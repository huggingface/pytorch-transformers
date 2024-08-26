# coding=utf-8
# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import glob
import itertools
import json
import os
import os.path
import sys
import tempfile
import threading
import unittest
import unittest.mock as mock
import uuid
import warnings
from pathlib import Path

import requests
from huggingface_hub import HfApi, HfFolder
from pytest import mark
from requests.exceptions import HTTPError

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    LlavaForConditionalGeneration,
    OwlViTForObjectDetection,
    PretrainedConfig,
    is_torch_available,
    logging,
)
from transformers.testing_utils import (
    TOKEN,
    CaptureLogger,
    LoggingLevel,
    TemporaryHubRepo,
    TestCasePlus,
    is_staging_test,
    require_accelerate,
    require_flax,
    require_safetensors,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_usr_bin_time,
    slow,
    torch_device,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_flax_available,
    is_tf_available,
    is_torch_sdpa_available,
    is_torchdynamo_available,
)


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig, NoSuperInitConfig  # noqa E402

if is_torch_available():
    import torch
    from safetensors.torch import save_file as safe_save_file
    from test_module.custom_modeling import CustomModel, NoSuperInitModel
    from torch import nn

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BertConfig,
        BertModel,
        CLIPTextModel,
        GenerationMixin,
        PreTrainedModel,
        T5Config,
        T5ForConditionalGeneration,
    )
    from transformers.modeling_attn_mask_utils import (
        AttentionMaskConverter,
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
        _prepare_4d_causal_attention_mask,
    )
    from transformers.modeling_utils import (
        _find_disjoint,
        _find_identical,
        dtype_byte_size,
    )
    from transformers.pytorch_utils import isin_mps_friendly

    # Fake pretrained models for tests
    class BaseModel(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = nn.Linear(5, 5)
            self.linear_2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear_2(self.linear(x))

    class BaseModelWithTiedWeights(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = nn.Linear(5, 5)
            self.linear_2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear_2(self.linear(x))

        def tie_weights(self):
            self.linear_2.weight = self.linear.weight

    class ModelWithHead(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            # linear is a common name between Base and Head on purpose.
            self.linear = nn.Linear(5, 5)
            self.linear2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear2(self.linear(self.base(x)))

    class ModelWithHeadAndTiedWeights(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            self.decoder = nn.Linear(5, 5)

        def forward(self, x):
            return self.decoder(self.base(x))

        def tie_weights(self):
            self.decoder.weight = self.base.linear.weight

    class Prepare4dCausalAttentionMaskModel(nn.Module):
        def forward(self, inputs_embeds):
            batch_size, seq_length, _ = inputs_embeds.shape
            past_key_values_length = 4
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
            return attention_mask

    class Create4dCausalAttentionMaskModel(nn.Module):
        def forward(self, inputs_embeds):
            batch_size, seq_length, _ = inputs_embeds.shape
            past_key_values_length = 4
            attention_mask = _create_4d_causal_attention_mask(
                (batch_size, seq_length),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            return attention_mask

    class Prepare4dAttentionMaskModel(nn.Module):
        def forward(self, mask, inputs_embeds):
            attention_mask = _prepare_4d_attention_mask(mask, dtype=inputs_embeds.dtype)
            return attention_mask

    class TestOffline(unittest.TestCase):
        def test_offline(self):
            # Ugly setup with monkeypatches, amending env vars here is too late as libs have already been imported
            from huggingface_hub import constants

            from transformers.utils import hub

            offlfine_env = hub._is_offline_mode
            hub_cache_env = constants.HF_HUB_CACHE
            hub_cache_env1 = constants.HUGGINGFACE_HUB_CACHE
            default_cache = constants.default_cache_path
            transformers_cache = hub.TRANSFORMERS_CACHE

            try:
                hub._is_offline_mode = True
                with tempfile.TemporaryDirectory() as tmpdir:
                    LOG.info("Temporary cache dir %s", tmpdir)
                    constants.HF_HUB_CACHE = tmpdir
                    constants.HUGGINGFACE_HUB_CACHE = tmpdir
                    constants.default_cache_path = tmpdir
                    hub.TRANSFORMERS_CACHE = tmpdir
                    # First offline load should fail
                    try:
                        AutoModelForImageClassification.from_pretrained(
                            TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None
                        )
                    except OSError:
                        LOG.info("Loading model %s in offline mode failed as expected", TINY_IMAGE_CLASSIF)
                    else:
                        self.fail("Loading model {} in offline mode should fail".format(TINY_IMAGE_CLASSIF))

                    # Download model -> Huggingface Hub not concerned by our offline mode
                    LOG.info("Downloading %s for offline tests", TINY_IMAGE_CLASSIF)
                    hub_api = HfApi()
                    local_dir = hub_api.snapshot_download(TINY_IMAGE_CLASSIF, cache_dir=tmpdir)

                    LOG.info("Model %s downloaded in %s", TINY_IMAGE_CLASSIF, local_dir)

                    AutoModelForImageClassification.from_pretrained(
                        TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None
                    )
            finally:
                # Tear down: reset env as it was before calling this test
                hub._is_offline_mode = offlfine_env
                constants.HF_HUB_CACHE = hub_cache_env
                constants.HUGGINGFACE_HUB_CACHE = hub_cache_env1
                constants.default_cache_path = default_cache
                hub.TRANSFORMERS_CACHE = transformers_cache

        def test_local_files_only(self):
            # Ugly setup with monkeypatches, amending env vars here is too late as libs have already been imported
            from huggingface_hub import constants

            from transformers.utils import hub

            hub_cache_env = constants.HF_HUB_CACHE
            hub_cache_env1 = constants.HUGGINGFACE_HUB_CACHE
            default_cache = constants.default_cache_path
            transformers_cache = hub.TRANSFORMERS_CACHE
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    LOG.info("Temporary cache dir %s", tmpdir)
                    constants.HF_HUB_CACHE = tmpdir
                    constants.HUGGINGFACE_HUB_CACHE = tmpdir
                    constants.default_cache_path = tmpdir
                    hub.TRANSFORMERS_CACHE = tmpdir
                    try:
                        AutoModelForImageClassification.from_pretrained(
                            TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None, local_files_only=True
                        )
                    except OSError:
                        LOG.info("Loading model %s in offline mode failed as expected", TINY_IMAGE_CLASSIF)
                    else:
                        self.fail("Loading model {} in offline mode should fail".format(TINY_IMAGE_CLASSIF))

                    LOG.info("Downloading %s for offline tests", TINY_IMAGE_CLASSIF)
                    hub_api = HfApi()
                    local_dir = hub_api.snapshot_download(TINY_IMAGE_CLASSIF, cache_dir=tmpdir)

                    LOG.info("Model %s downloaded in %s", TINY_IMAGE_CLASSIF, local_dir)

                    AutoModelForImageClassification.from_pretrained(
                        TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None, local_files_only=True
                    )
            finally:
                # Tear down: reset env as it was before calling this test
                constants.HF_HUB_CACHE = hub_cache_env
                constants.HUGGINGFACE_HUB_CACHE = hub_cache_env1
                constants.default_cache_path = default_cache
                hub.TRANSFORMERS_CACHE = transformers_cache


if is_flax_available():
    from transformers import FlaxBertModel

if is_tf_available():
    from transformers import TFBertModel


TINY_T5 = "patrickvonplaten/t5-tiny-random"
TINY_BERT_FOR_TOKEN_CLASSIFICATION = "hf-internal-testing/tiny-bert-for-token-classification"
TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"
TINY_IMAGE_CLASSIF = "hf-internal-testing/tiny-random-SiglipForImageClassification"
TINY_LLAVA = "hf-internal-testing/tiny-random-LlavaForConditionalGeneration"

LOG = logging.get_logger(__name__)


def check_models_equal(model1, model2):
    models_are_equal = True
    for model1_p, model2_p in zip(model1.parameters(), model2.parameters()):
        if model1_p.data.ne(model2_p.data).sum() > 0:
            models_are_equal = False

    return models_are_equal


@require_torch
class ModelUtilsTest(TestCasePlus):
    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = BertConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, PretrainedConfig)

        model = BertModel.from_pretrained(model_name)
        model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, PreTrainedModel)

        self.assertEqual(len(loading_info["missing_keys"]), 0)
        self.assertEqual(len(loading_info["unexpected_keys"]), 8)
        self.assertEqual(len(loading_info["mismatched_keys"]), 0)
        self.assertEqual(len(loading_info["error_msgs"]), 0)

        config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

        # Not sure this is the intended behavior. TODO fix Lysandre & Thom
        config.name_or_path = model_name

        model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self.assertEqual(model.config.output_hidden_states, True)
        self.assertEqual(model.config, config)

    def test_model_from_pretrained_subfolder(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder))

            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)

            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_manually_shared_disjointed_tensors_optimum(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        # Let's fuse qkv
        attn = model.encoder.layer[0].attention.self
        q = attn.query.weight
        k = attn.key.weight
        v = attn.value.weight
        # Force some shared storage
        qkv = torch.stack([q, k, v], dim=0)
        attn.query.weight = torch.nn.Parameter(qkv[0])
        attn.key.weight = torch.nn.Parameter(qkv[1])
        attn.value.weight = torch.nn.Parameter(qkv[2])
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model_loaded = BertModel.from_pretrained(tmp_dir)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_subfolder_sharded(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder), max_shard_size="10KB")

            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)

            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_hub_subfolder(self):
        subfolder = "bert"
        model_id = "hf-internal-testing/tiny-random-bert-subfolder"
        with self.assertRaises(OSError):
            _ = BertModel.from_pretrained(model_id)

        model = BertModel.from_pretrained(model_id, subfolder=subfolder)

        self.assertIsNotNone(model)

    def test_model_from_pretrained_hub_subfolder_sharded(self):
        subfolder = "bert"
        model_id = "hf-internal-testing/tiny-random-bert-sharded-subfolder"
        with self.assertRaises(OSError):
            _ = BertModel.from_pretrained(model_id)

        model = BertModel.from_pretrained(model_id, subfolder=subfolder)

        self.assertIsNotNone(model)

    def test_model_from_pretrained_with_different_pretrained_model_name(self):
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
        self.assertIsNotNone(model)

        logger = logging.get_logger("transformers.configuration_utils")
        with LoggingLevel(logging.WARNING):
            with CaptureLogger(logger) as cl:
                BertModel.from_pretrained(TINY_T5)
        self.assertTrue("You are using a model of type t5 to instantiate a model of type bert" in cl.out)

    @require_accelerate
    def test_model_from_pretrained_with_none_quantization_config(self):
        # Needs a device_map for to enter the low_cpu_mem branch. We also load AutoModelForSequenceClassification
        # deliberately to enter the missing keys branch.
        model = AutoModelForSequenceClassification.from_pretrained(
            TINY_MISTRAL, device_map="auto", quantization_config=None
        )
        self.assertIsNotNone(model)

    def test_model_from_config_torch_dtype(self):
        # test that the model can be instantiated with dtype of user's choice - as long as it's a
        # float dtype. To make it happen config.torch_dtype needs to be set before instantiating the
        # model from the config object.

        config = T5Config.from_pretrained(TINY_T5)
        model = AutoModel.from_config(config)
        # XXX: isn't supported
        # model = T5ForConditionalGeneration.from_config(config)
        self.assertEqual(model.dtype, torch.float32)

        model = AutoModel.from_config(config, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # torch.set_default_dtype() supports only float dtypes, so will fail with non-float type
        with self.assertRaises(ValueError):
            model = AutoModel.from_config(config, torch_dtype=torch.int64)

    def test_model_from_config_torch_dtype_str(self):
        # test that from_pretrained works with torch_dtype being strings like "float32" for PyTorch backend
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype="float32")
        self.assertEqual(model.dtype, torch.float32)

        model = AutoModel.from_pretrained(TINY_T5, torch_dtype="float16")
        self.assertEqual(model.dtype, torch.float16)

        # torch.set_default_dtype() supports only float dtypes, so will fail with non-float type
        with self.assertRaises(ValueError):
            model = AutoModel.from_pretrained(TINY_T5, torch_dtype="int64")

    def test_model_from_config_torch_dtype_composite(self):
        """
        Test that from_pretrained works with torch_dtype being as a dict per each sub-config in composite config
        """
        # should be able to set torch_dtype as a simple string and the model loads it correctly
        model = LlavaForConditionalGeneration.from_pretrained(TINY_LLAVA, torch_dtype="float32")
        self.assertEqual(model.language_model.dtype, torch.float32)
        self.assertEqual(model.vision_tower.dtype, torch.float32)

        model = LlavaForConditionalGeneration.from_pretrained(TINY_LLAVA, torch_dtype="float16")
        self.assertEqual(model.language_model.dtype, torch.float16)
        self.assertEqual(model.vision_tower.dtype, torch.float16)

        # should be able to set torch_dtype as a dict for each sub-config
        model = LlavaForConditionalGeneration.from_pretrained(
            TINY_LLAVA, torch_dtype={"text_config": "float32", "vision_config": "float16", "": "bfloat16"}
        )
        self.assertEqual(model.language_model.dtype, torch.float32)
        self.assertEqual(model.vision_tower.dtype, torch.float16)
        self.assertEqual(model.multi_modal_projector.linear_1.weight.dtype, torch.bfloat16)

        # should be able to set the values as torch.dtype (not str)
        model = LlavaForConditionalGeneration.from_pretrained(
            TINY_LLAVA, torch_dtype={"text_config": torch.float32, "vision_config": torch.float16, "": torch.bfloat16}
        )
        self.assertEqual(model.language_model.dtype, torch.float32)
        self.assertEqual(model.vision_tower.dtype, torch.float16)
        self.assertEqual(model.multi_modal_projector.linear_1.weight.dtype, torch.bfloat16)

        # should be able to set the values in configs directly and pass it to `from_pretrained`
        config = copy.deepcopy(model.config)
        config.text_config.torch_dtype = torch.float32
        config.vision_config.torch_dtype = torch.bfloat16
        config.torch_dtype = torch.float16
        model = LlavaForConditionalGeneration.from_pretrained(TINY_LLAVA, config=config, torch_dtype="auto")
        self.assertEqual(model.language_model.dtype, torch.float32)
        self.assertEqual(model.vision_tower.dtype, torch.bfloat16)
        self.assertEqual(model.multi_modal_projector.linear_1.weight.dtype, torch.float16)

        # but if the model has `_keep_in_fp32_modules` then those modules should be in fp32 no matter what
        LlavaForConditionalGeneration._keep_in_fp32_modules = ["multi_modal_projector"]
        model = LlavaForConditionalGeneration.from_pretrained(TINY_LLAVA, config=config, torch_dtype="auto")
        self.assertEqual(model.language_model.dtype, torch.float32)
        self.assertEqual(model.vision_tower.dtype, torch.bfloat16)
        self.assertEqual(model.multi_modal_projector.linear_1.weight.dtype, torch.float32)

        # torch.set_default_dtype() supports only float dtypes, so will fail with non-float type
        with self.assertRaises(ValueError):
            model = LlavaForConditionalGeneration.from_pretrained(TINY_LLAVA, torch_dtype="int64")
            model = LlavaForConditionalGeneration.from_pretrained(
                TINY_LLAVA, torch_dtype={"text_config": "float32", "vision_config": "int64", "": "float16"}
            )

    @require_torch
    def test_model_from_pretrained_meta_device(self):
        def is_on_meta(model_id, dtype):
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
                return all(value.device.type == "meta" for value in model.state_dict().values())

        model_ids = ("fxmarty/tiny-llama-fast-tokenizer", "fxmarty/small-llama-testing")
        dtypes = (None, "auto", torch.float16)

        for model_id, dtype in itertools.product(model_ids, dtypes):
            self.assertTrue(is_on_meta(model_id, dtype))

    def test_model_from_pretrained_torch_dtype(self):
        # test that the model can be instantiated with dtype of either
        # 1. explicit from_pretrained's torch_dtype argument
        # 2. via autodiscovery by looking at model weights (torch_dtype="auto")
        # so if a model.half() was saved, we want it to be instantiated as such.
        #
        # test an explicit model class, but also AutoModel separately as the latter goes through a different code path
        model_path = self.get_auto_remove_tmp_dir()

        # baseline - we know TINY_T5 is fp32 model
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
        self.assertEqual(model.dtype, torch.float32)

        def remove_torch_dtype(model_path):
            file = f"{model_path}/config.json"
            with open(file, "r", encoding="utf-8") as f:
                s = json.load(f)
            s.pop("torch_dtype")
            with open(file, "w", encoding="utf-8") as f:
                json.dump(s, f)

        # test the default fp32 save_pretrained => from_pretrained cycle
        model.save_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.assertEqual(model.dtype, torch.float32)
        # 1. test torch_dtype="auto" via `config.torch_dtype`
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float32)
        # 2. test torch_dtype="auto" via auto-derivation
        # now remove the torch_dtype entry from config.json and try "auto" again which should
        # perform auto-derivation from weights
        remove_torch_dtype(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float32)

        # test forced loading in fp16 (even though the weights are in fp32)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # test fp16 save_pretrained, loaded with auto-detection
        model = model.half()
        model.save_pretrained(model_path)
        # 1. test torch_dtype="auto" via `config.torch_dtype`
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.config.torch_dtype, torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        # tests `config.torch_dtype` saving
        with open(f"{model_path}/config.json") as f:
            config_dict = json.load(f)
        self.assertEqual(config_dict["torch_dtype"], "float16")
        # 2. test torch_dtype="auto" via auto-derivation
        # now same with using config info
        remove_torch_dtype(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float16)

        # 3. now retest that AutoModel behaves the same wrt torch_dtype="auto" as T5ForConditionalGeneration
        model = AutoModel.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float16)

        # test fp16 save_pretrained, loaded with the explicit fp16
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # test AutoModel separately as it goes through a different path
        # test auto-detection - as currently TINY_T5 doesn't have torch_dtype entry
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype="auto")
        # test that the config object didn't get polluted with torch_dtype="auto"
        # there was a bug that after this call we ended up with config.torch_dtype=="auto"
        self.assertNotEqual(model.config.torch_dtype, "auto")
        # now test the outcome
        self.assertEqual(model.dtype, torch.float32)
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # test model whose first param is not of a floating type, but int
        model = AutoModel.from_pretrained(TINY_BERT_FOR_TOKEN_CLASSIFICATION, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float32)

    def test_model_from_pretrained_attn_implementation(self):
        # test that the model can be instantiated with attn_implementation of either
        # 1. explicit from_pretrained's attn_implementation argument
        # 2. explicit from_pretrained's attn_implementation argument with a config argument
        attn_implementation_available = ["eager"]
        if is_torch_sdpa_available():
            attn_implementation_available.append("sdpa")

        if is_flash_attn_2_available():
            attn_implementation_available.append("flash_attention_2")

        for requested_attn_implementation in attn_implementation_available:
            model = AutoModelForCausalLM.from_pretrained(
                TINY_MISTRAL, attn_implementation=requested_attn_implementation
            )
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

            config = AutoConfig.from_pretrained(TINY_MISTRAL)
            model = AutoModelForCausalLM.from_pretrained(
                TINY_MISTRAL, config=config, attn_implementation=requested_attn_implementation
            )
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

    def test_model_from_config_attn_implementation(self):
        # test that the model can be instantiated with attn_implementation of either
        # 1. config created with explicit attn_implementatation and from_config
        # 2. explicit from_config's attn_implementation argument with a config argument
        # 3. config created with explicit attn_implementatation and from_config overriding with explicit attn_implementation argument
        attn_implementation_available = ["eager"]
        if is_torch_sdpa_available():
            attn_implementation_available.append("sdpa")

        if is_flash_attn_2_available():
            attn_implementation_available.append("flash_attention_2")

        for requested_attn_implementation in attn_implementation_available:
            config = AutoConfig.from_pretrained(TINY_MISTRAL, attn_implementation=requested_attn_implementation)
            # Ensure the config was set correctly
            self.assertEqual(config._attn_implementation, requested_attn_implementation)
            self.assertEqual(config._attn_implementation_internal, requested_attn_implementation)
            model = AutoModelForCausalLM.from_config(config)
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

            config = AutoConfig.from_pretrained(TINY_MISTRAL)
            # When the config is not set, the default is "eager"
            self.assertEqual(config._attn_implementation, "eager")
            self.assertEqual(config._attn_implementation_internal, None)
            model = AutoModelForCausalLM.from_config(config=config, attn_implementation=requested_attn_implementation)
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

            # Set a nonsense attn_implementation in the config, which should be overridden by the explicit argument
            config = AutoConfig.from_pretrained(TINY_MISTRAL, attn_implementation="foo-bar-baz")
            self.assertEqual(config._attn_implementation, "foo-bar-baz")
            self.assertEqual(config._attn_implementation_internal, "foo-bar-baz")
            model = AutoModelForCausalLM.from_config(config=config, attn_implementation=requested_attn_implementation)
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

    def test_torch_dtype_byte_sizes(self):
        torch_dtypes_and_bytes = [
            (torch.double, 8),
            (torch.float64, 8),
            (torch.float, 4),
            (torch.float32, 4),
            (torch.half, 2),
            (torch.float16, 2),
            (torch.bfloat16, 2),
            (torch.long, 8),
            (torch.int64, 8),
            (torch.int, 4),
            (torch.int32, 4),
            (torch.short, 2),
            (torch.int16, 2),
            (torch.uint8, 1),
            (torch.int8, 1),
            (torch.float8_e4m3fn, 1),
            (torch.float8_e5m2, 1),
            (torch.bool, 0.125),
        ]

        for torch_dtype, bytes_per_element in torch_dtypes_and_bytes:
            self.assertEqual(dtype_byte_size(torch_dtype), bytes_per_element)

    def test_no_super_init_config_and_model(self):
        config = NoSuperInitConfig(attribute=32)
        model = NoSuperInitModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            new_model = NoSuperInitModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_checkpoint_sharding_local_bin(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
            for max_size in ["50kB", "100kB", "200kB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size, safe_serialization=False)

                # Get each shard file and its size
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith(".bin"):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)

                index_file = os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)
                # Check there is an index but no regular weight file
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

                # Check a file is bigger than max_size only when it has a single weight
                for shard_file, size in shard_to_size.items():
                    max_size_int = int(max_size[:-2]) * 10**3
                    # Note: pickle adds some junk so the weight of the file can end up being slightly bigger than
                    # the size asked for (since we count parameters)
                    if size >= max_size_int + 50000:
                        state_dict = torch.load(shard_file)
                        self.assertEqual(len(state_dict), 1)

                # Check the index and the shard files found match
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.loads(f.read())

                all_shards = set(index["weight_map"].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith(".bin")}
                self.assertSetEqual(all_shards, shards_found)

                # Finally, check the model can be reloaded
                new_model = BertModel.from_pretrained(tmp_dir)
                for p1, p2 in zip(model.parameters(), new_model.parameters()):
                    self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_sharding_from_hub(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
        # the model above is the same as the model below, just a sharded version.
        ref_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        for p1, p2 in zip(model.parameters(), ref_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local_bin(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=False)

            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local_sharded_bin(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB", safe_serialization=False)

            weights_index_name = ".".join(WEIGHTS_INDEX_NAME.split(".")[:-1] + ["v2"] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))

            for i in range(1, 5):
                weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + [f"v2-0000{i}-of-00005"] + ["bin"])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_checkpoint_variant_local_safe(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=True)

            weights_name = ".".join(SAFE_WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["safetensors"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_checkpoint_variant_local_sharded_safe(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB", safe_serialization=True)

            weights_index_name = ".".join(SAFE_WEIGHTS_INDEX_NAME.split(".")[:-1] + ["v2"] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            for i in range(1, 5):
                weights_name = ".".join(SAFE_WEIGHTS_NAME.split(".")[:-1] + [f"v2-0000{i}-of-00005"] + ["safetensors"])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_loading_only_safetensors_available(self):
        # Test that the loading behaviour is as expected when only safetensor checkpoints are available
        # - We can load the model with use_safetensors=True
        # - We can load the model without specifying use_safetensors i.e. we search for the available checkpoint,
        #   preferring safetensors
        # - We cannot load the model with use_safetensors=False
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, max_shard_size="50kB", safe_serialization=True)

            weights_index_name = ".".join(SAFE_WEIGHTS_INDEX_NAME.split(".")[:-1] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))

            for i in range(1, 5):
                weights_name = f"model-0000{i}-of-00005" + ".safetensors"
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            # Setting use_safetensors=False should raise an error as the checkpoint was saved with safetensors=True
            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir, use_safetensors=False)

            # We can load the model with use_safetensors=True
            new_model = BertModel.from_pretrained(tmp_dir, use_safetensors=True)

            # We can load the model without specifying use_safetensors
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_loading_only_pytorch_bin_available(self):
        # Test that the loading behaviour is as expected when only pytorch checkpoints are available
        # - We can load the model with use_safetensors=False
        # - We can load the model without specifying use_safetensors i.e. we search for the available checkpoint,
        #   preferring safetensors but falling back to pytorch
        # - We cannot load the model with use_safetensors=True
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, max_shard_size="50kB", safe_serialization=False)

            weights_index_name = ".".join(WEIGHTS_INDEX_NAME.split(".")[:-1] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))

            for i in range(1, 5):
                weights_name = WEIGHTS_NAME.split(".")[0].split("_")[0] + f"_model-0000{i}-of-00005" + ".bin"
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            # Setting use_safetensors=True should raise an error as the checkpoint was saved with safetensors=False
            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir, use_safetensors=True)

            # We can load the model with use_safetensors=False
            new_model = BertModel.from_pretrained(tmp_dir, use_safetensors=False)

            # We can load the model without specifying use_safetensors
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_hub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    def test_checkpoint_variant_hub_sharded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_sharded_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    def test_checkpoint_variant_save_load_bin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=False)
            # saving will create a variant checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            model.save_pretrained(tmp_dir, safe_serialization=False)
            # saving shouldn't delete variant checkpoints
            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            # there should be a normal checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

        self.assertIsNotNone(model)

    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_functional(self):
        # test that we can use `from_pretrained(..., low_cpu_mem_usage=True)` with normal and
        # sharded models

        mnames = [
            "hf-internal-testing/tiny-random-bert-sharded",
            "hf-internal-testing/tiny-random-bert",
        ]
        for mname in mnames:
            _ = BertModel.from_pretrained(mname, low_cpu_mem_usage=True)

    @require_usr_bin_time
    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_equal(self):
        # Before this would test that `from_pretrained(..., low_cpu_mem_usage=True)` uses less cpu memory than default
        # Now though these should be around the same.
        # TODO: Look for good bounds to check that their timings are near the same

        mname = "hf-internal-testing/tiny-random-bert"

        preamble = "from transformers import AutoModel"
        one_liner_str = f'{preamble}; AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=False)'
        # Save this output as `max_rss_normal` if testing memory results
        max_rss_normal = self.python_one_liner_max_rss(one_liner_str)
        # print(f"{max_rss_normal=}")

        one_liner_str = f'{preamble};  AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=True)'
        # Save this output as `max_rss_low_mem` if testing memory results
        max_rss_low_mem = self.python_one_liner_max_rss(one_liner_str)

        # Should be within 2MBs of each other (overhead)
        self.assertAlmostEqual(
            max_rss_normal / 1024 / 1024,
            max_rss_low_mem / 1024 / 1024,
            delta=2,
            msg="using `low_cpu_mem_usage` should incur the same memory usage in both cases.",
        )

        # if you want to compare things manually, let's first look at the size of the model in bytes
        # model = BertModel.from_pretrained(mname, low_cpu_mem_usage=False)
        # total_numel = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        # total_bytes = total_numel * 4  # 420MB
        # Now the diff_bytes should be very close to total_bytes, but the reports are inconsistent.
        # The easiest way to test this is to switch the model and torch.load to do all the work on
        # gpu - that way one can measure exactly the total and peak memory used. Perhaps once we add
        # functionality to load models directly on gpu, this test can be rewritten to use torch's
        # cuda memory tracking and then we should be able to do a much more precise test.

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_multi_accelerator
    @slow
    def test_model_parallelism_gpt2(self):
        device_map = {"transformer.wte": 0, "transformer.wpe": 0, "lm_head": 0, "transformer.ln_f": 1}
        for i in range(12):
            device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1

        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        output = model.generate(inputs["input_ids"].to(f"{torch_device}:0"))

        text_output = tokenizer.decode(output[0].tolist())
        self.assertEqual(text_output, "Hello, my name is John. I'm a writer, and I'm a writer. I'm")

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_from_pretrained_disk_offload_task_model(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        device_map = {
            "transformer.wte": f"{torch_device}:0",
            "transformer.wpe": f"{torch_device}:0",
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": "cpu",
            "transformer.h.3": "disk",
            "transformer.h.4": "disk",
            "transformer.ln_f": f"{torch_device}:0",
            "lm_head": f"{torch_device}:0",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = torch.tensor([[1, 2, 3]]).to(f"{torch_device}:0")

            model.save_pretrained(tmp_dir)
            new_model = AutoModelForCausalLM.from_pretrained(tmp_dir).to(f"{torch_device}:0")
            outputs1 = new_model.to(f"{torch_device}:0")(inputs)

            offload_folder = os.path.join(tmp_dir, "offload")
            new_model_with_offload = AutoModelForCausalLM.from_pretrained(
                tmp_dir, device_map=device_map, offload_folder=offload_folder
            )
            outputs2 = new_model_with_offload(inputs)

            self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

            # With state dict temp offload
            new_model_with_offload = AutoModelForCausalLM.from_pretrained(
                tmp_dir,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
            outputs2 = new_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_from_pretrained_disk_offload_derived_to_base_model(self):
        derived_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        device_map = {
            "wte": f"{torch_device}:0",
            "wpe": f"{torch_device}:0",
            "h.0": "cpu",
            "h.1": "cpu",
            "h.2": "cpu",
            "h.3": "disk",
            "h.4": "disk",
            "ln_f": f"{torch_device}:0",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = torch.tensor([[1, 2, 3]]).to(f"{torch_device}:0")
            derived_model.save_pretrained(tmp_dir, use_safetensors=True)
            base_model = AutoModel.from_pretrained(tmp_dir)
            outputs1 = base_model.to(f"{torch_device}:0")(inputs)

            # with disk offload
            offload_folder = os.path.join(tmp_dir, "offload")
            base_model_with_offload = AutoModel.from_pretrained(
                tmp_dir, device_map=device_map, offload_folder=offload_folder
            )
            outputs2 = base_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1[0].cpu(), outputs2[0].cpu()))

            # With state dict temp offload
            new_model_with_offload = AutoModel.from_pretrained(
                tmp_dir,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
            outputs2 = new_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1[0].cpu(), outputs2[0].cpu()))

    @slow
    @require_torch
    def test_from_pretrained_non_contiguous_checkpoint(self):
        # See: https://github.com/huggingface/transformers/pull/28414
        # Tiny models on the Hub have contiguous weights, contrarily to google/owlvit
        model = OwlViTForObjectDetection.from_pretrained("fxmarty/owlvit-tiny-non-contiguous-weight")
        self.assertTrue(model.owlvit.visual_projection.weight.is_contiguous())

        model = OwlViTForObjectDetection.from_pretrained(
            "fxmarty/owlvit-tiny-non-contiguous-weight", device_map="auto"
        )
        self.assertTrue(model.owlvit.visual_projection.weight.is_contiguous())

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=False)
            model.save_pretrained(tmp_dir, safe_serialization=True)

    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()

    @require_accelerate
    @mark.accelerate_tests
    def test_save_model_with_device_map_cpu(self):
        model_id = "hf-internal-testing/tiny-random-gpt2"
        inputs = torch.tensor([[1, 2, 3]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
            output = model(inputs)[0]
            model.save_pretrained(
                tmp_dir, max_shard_size="200KB"
            )  # model is 1.6MB, max shard size is allocated to cpu by default
            saved_model = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map="cpu")
            saved_model_output = saved_model(inputs)[0]

        self.assertTrue(torch.allclose(output, saved_model_output))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_save_offloaded_model(self):
        device_map = {
            "transformer.wte": f"{torch_device}:0",
            "transformer.wpe": f"{torch_device}:0",
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": "cpu",
            "transformer.h.3": "disk",
            "transformer.h.4": "disk",
            "transformer.ln_f": f"{torch_device}:0",
            "lm_head": f"{torch_device}:0",
        }

        # check_models_equal requires onloaded tensors
        model_id = "hf-internal-testing/tiny-random-gpt2"
        onloaded_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu").to(f"{torch_device}:0")
        inputs = torch.tensor([[1, 2, 3]]).to(f"{torch_device}:0")
        output = onloaded_model(inputs)[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            offload_folder = os.path.join(tmp_dir, "offload")
            offloaded_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, offload_folder=offload_folder
            )
            presaved_output = offloaded_model(inputs)[0]
            offloaded_model.save_pretrained(
                tmp_dir, max_shard_size="200KB"
            )  # model is 1.6MB, max shard size is allocated to cpu by default
            saved_model = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map)
            postsaved_output = saved_model(inputs)[0]

        self.assertTrue(torch.allclose(output, presaved_output, atol=1e-4))
        self.assertTrue(torch.allclose(presaved_output, postsaved_output))

    @require_safetensors
    def test_use_safetensors(self):
        # Should not raise anymore
        AutoModel.from_pretrained("hf-internal-testing/tiny-random-RobertaModel", use_safetensors=True)

        # test that error if only safetensors is available
        with self.assertRaises(OSError) as env_error:
            BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors", use_safetensors=False)

        self.assertTrue("does not appear to have a file named pytorch_model.bin" in str(env_error.exception))

        # test that only safetensors if both available and use_safetensors=False
        with tempfile.TemporaryDirectory() as tmp_dir:
            CLIPTextModel.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all",
                subfolder="text_encoder",
                use_safetensors=False,
                cache_dir=tmp_dir,
            )

            all_downloaded_files = glob.glob(os.path.join(tmp_dir, "*", "snapshots", "*", "*", "*"))
            self.assertTrue(any(f.endswith("bin") for f in all_downloaded_files))
            self.assertFalse(any(f.endswith("safetensors") for f in all_downloaded_files))

        # test that no safetensors if both available and use_safetensors=True
        with tempfile.TemporaryDirectory() as tmp_dir:
            CLIPTextModel.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all",
                subfolder="text_encoder",
                use_safetensors=True,
                cache_dir=tmp_dir,
            )

            all_downloaded_files = glob.glob(os.path.join(tmp_dir, "*", "snapshots", "*", "*", "*"))
            self.assertTrue(any(f.endswith("safetensors") for f in all_downloaded_files))
            self.assertFalse(any(f.endswith("bin") for f in all_downloaded_files))

        # test no model file found when use_safetensors=None (default when safetensors package available)
        with self.assertRaises(OSError) as missing_model_file_error:
            BertModel.from_pretrained("hf-internal-testing/config-no-model")

        self.assertTrue(
            "does not appear to have a file named pytorch_model.bin, model.safetensors,"
            in str(missing_model_file_error.exception)
        )

        with self.assertRaises(OSError) as missing_model_file_error:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                    f.write("{}")
                f.close()
                BertModel.from_pretrained(tmp_dir)

        self.assertTrue(
            "Error no file named pytorch_model.bin, model.safetensors" in str(missing_model_file_error.exception)
        )

    @require_safetensors
    def test_safetensors_save_and_load(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            # No pytorch_model.bin file, only a model.safetensors
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

            new_model = BertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_load_from_hub(self):
        safetensors_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors")
        pytorch_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Check models are equal
        for p1, p2 in zip(safetensors_model.parameters(), pytorch_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_save_and_load_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="100kB")
            # No pytorch_model.bin index file, only a model.safetensors index
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
            # No regular weights file
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

            new_model = BertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_load_from_hub_sharded(self):
        safetensors_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded-safetensors")
        pytorch_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")

        # Check models are equal
        for p1, p2 in zip(safetensors_model.parameters(), pytorch_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_base_model_to_head_model_load(self):
        base_model = BaseModel(PretrainedConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_model.save_pretrained(tmp_dir, safe_serialization=False)

            # Can load a base model in a model with head
            model = ModelWithHead.from_pretrained(tmp_dir)
            for p1, p2 in zip(model.base.parameters(), base_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

            # It doesn't work if the state dict has a mix of keys of the head and base without prefix though.
            base_state_dict = base_model.state_dict()
            head_state_dict = model.state_dict()
            base_state_dict["linear2.weight"] = head_state_dict["linear2.weight"]
            base_state_dict["linear2.bias"] = head_state_dict["linear2.bias"]
            safe_save_file(base_state_dict, os.path.join(tmp_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})

            with self.assertRaisesRegex(
                ValueError, "The state dictionary of the model you are trying to load is corrupted."
            ):
                _ = ModelWithHead.from_pretrained(tmp_dir)

    def test_tied_weights_reload(self):
        # Base
        model = BaseModelWithTiedWeights(PretrainedConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            new_model = BaseModelWithTiedWeights.from_pretrained(tmp_dir)
            self.assertIs(new_model.linear.weight, new_model.linear_2.weight)

            state_dict = model.state_dict()
            # Remove tied weight from state_dict -> model should load with no complain of missing keys
            del state_dict["linear_2.weight"]
            torch.save(state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))
            new_model, load_info = BaseModelWithTiedWeights.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertListEqual(load_info["missing_keys"], [])
            self.assertIs(new_model.linear.weight, new_model.linear_2.weight)

            # With head
            model.save_pretrained(tmp_dir)
            new_model, load_info = ModelWithHeadAndTiedWeights.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIs(new_model.base.linear.weight, new_model.decoder.weight)
            # Should only complain about the missing bias
            self.assertListEqual(load_info["missing_keys"], ["decoder.bias"])

    def test_unexpected_keys_warnings(self):
        model = ModelWithHead(PretrainedConfig())
        logger = logging.get_logger("transformers.modeling_utils")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            # Loading the model with a new class, we don't get a warning for unexpected weights, just an info
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    _, loading_info = BaseModel.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertNotIn("were not used when initializing ModelWithHead", cl.out)
            self.assertEqual(
                set(loading_info["unexpected_keys"]),
                {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias"},
            )

            # Loading the model with the same class, we do get a warning for unexpected weights
            state_dict = model.state_dict()
            state_dict["added_key"] = copy.deepcopy(state_dict["linear.weight"])
            safe_save_file(state_dict, os.path.join(tmp_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    _, loading_info = ModelWithHead.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIn("were not used when initializing ModelWithHead: ['added_key']", cl.out)
            self.assertEqual(loading_info["unexpected_keys"], ["added_key"])

    def test_warn_if_padding_and_no_attention_mask(self):
        logger = logging.get_logger("transformers.modeling_utils")

        with self.subTest("Ensure no warnings when pad_token_id is None."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config_no_pad_token = PretrainedConfig()
                    config_no_pad_token.pad_token_id = None
                    model = ModelWithHead(config_no_pad_token)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure no warnings when there is an attention_mask."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure no warnings when there are no pad_token_ids in the input_ids."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[1, 345, 232, 328, 740, 140, 1695, 69, 6078, 2341, 25]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure a warning is shown when the input_ids start with a pad_token_id."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure a warning is shown when the input_ids end with a pad_token_id."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[432, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure that the warning is shown at most once."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertEqual(cl.out.count("We strongly recommend passing in an `attention_mask`"), 1)

        with self.subTest("Ensure a different warning is shown when the pad_token_id is equal to the bos_token_id."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    config.bos_token_id = config.pad_token_id
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("You may ignore this warning if your `pad_token_id`", cl.out)

        if not is_torchdynamo_available():
            self.skipTest(reason="torchdynamo is not available")
        with self.subTest("Ensure that the warning code is skipped when compiling with torchdynamo."):
            logger.warning_once.cache_clear()
            from torch._dynamo import config, testing

            config = PretrainedConfig()
            config.pad_token_id = 0
            model = ModelWithHead(config)
            input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])

            def f(input_ids):
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)

            compile_counter = testing.CompileCounter()
            opt_fn = torch.compile(f, dynamic=True, backend=compile_counter)
            opt_fn(input_ids)
            self.assertEqual(compile_counter.frame_count, 0)

    @require_torch_accelerator
    @slow
    def test_pretrained_low_mem_new_config(self):
        # Checking for 1 model(the same one which was described in the issue) .
        model_ids = ["openai-community/gpt2"]

        for model_id in model_ids:
            model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
            model_config.n_layer = 48
            model_config.n_head = 25
            model_config.n_embd = 1600
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=model_config,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model_ref = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)

            self.assertEqual(model.__class__.__name__, model_ref.__class__.__name__)

    def test_generation_config_is_loaded_with_model(self):
        # Note: `hf-internal-testing/tiny-random-MistralForCausalLM` has a `generation_config.json`
        # containing `bos_token_id: 1`

        # 1. Load without further parameters
        model = AutoModelForCausalLM.from_pretrained(TINY_MISTRAL)
        self.assertEqual(model.generation_config.bos_token_id, 1)

        # 2. Load with `device_map`
        model = AutoModelForCausalLM.from_pretrained(TINY_MISTRAL, device_map="auto")
        self.assertEqual(model.generation_config.bos_token_id, 1)

    @require_safetensors
    def test_safetensors_torch_from_torch(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_safetensors
    @require_flax
    def test_safetensors_torch_from_flax(self):
        hub_model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")
        model = FlaxBertModel.from_pretrained("hf-internal-testing/tiny-bert-flax-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(hub_model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_tf
    @require_safetensors
    def test_safetensors_torch_from_tf(self):
        hub_model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-tf-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(hub_model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_safetensors
    def test_safetensors_torch_from_torch_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="100kB")
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_modifying_model_config_gets_moved_to_generation_config(self):
        """
        Calling `model.save_pretrained` should move the changes made to `generate` parameterization in the model config
        to the generation config.
        """
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        # Initially, the repetition penalty has its default value in `model.config`. The `model.generation_config` will
        # have the exact same default
        self.assertTrue(model.config.repetition_penalty == 1.0)
        self.assertTrue(model.generation_config.repetition_penalty == 1.0)
        # If the user attempts to save a custom generation parameter:
        model.config.repetition_penalty = 3.0
        with warnings.catch_warnings(record=True) as warning_list:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                # 1 - That parameter will be removed from `model.config`. We don't want to use `model.config` to store
                # generative parameters, and the old default (1.0) would no longer relect the user's wishes.
                self.assertTrue(model.config.repetition_penalty is None)
                # 2 - That parameter will be set in `model.generation_config` instead.
                self.assertTrue(model.generation_config.repetition_penalty == 3.0)
        # 3 - The user will see a warning regarding the custom parameter that has been moved.
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue("Moving the following attributes" in str(warning_list[0].message))
        self.assertTrue("repetition_penalty" in str(warning_list[0].message))

    @require_safetensors
    def test_model_from_pretrained_from_mlx(self):
        from safetensors import safe_open

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-mistral-mlx")
        self.assertIsNotNone(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            with safe_open(os.path.join(tmp_dir, "model.safetensors"), framework="pt") as f:
                metadata = f.metadata()
                self.assertEqual(metadata.get("format"), "pt")
            new_model = AutoModelForCausalLM.from_pretrained(tmp_dir)

        input_ids = torch.randint(100, 1000, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
            outputs_from_saved = new_model(input_ids)
            self.assertTrue(torch.allclose(outputs_from_saved["logits"], outputs["logits"]))

    def test_warning_for_beta_gamma_parameters(self):
        class TestModelGamma(PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.gamma_param = nn.Parameter(torch.ones(10))
                self.post_init()

            def forward(self):
                return self.gamma_param.sum()

        logger = logging.get_logger("transformers.modeling_utils")
        config = PretrainedConfig()
        warning_msg_gamma = "`gamma_param` -> `weight_param`"
        model = TestModelGamma(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            with LoggingLevel(logging.INFO):
                with CaptureLogger(logger) as cl1:
                    _, loading_info = TestModelGamma.from_pretrained(tmp_dir, config=config, output_loading_info=True)

        missing_keys = loading_info["missing_keys"]
        unexpected_keys = loading_info["unexpected_keys"]
        self.assertIn("`TestModelGamma`", cl1.out)
        self.assertIn(warning_msg_gamma, cl1.out)
        self.assertIn("gamma_param", missing_keys)
        self.assertIn("weight_param", unexpected_keys)

        class TestModelBeta(PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.beta_param = nn.Parameter(torch.ones(10))
                self.post_init()

            def forward(self):
                return self.beta_param.sum()

        warning_msg_beta = "`beta_param` -> `bias_param`"
        model = TestModelBeta(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            with LoggingLevel(logging.INFO):
                with CaptureLogger(logger) as cl2:
                    _, loading_info = TestModelBeta.from_pretrained(tmp_dir, config=config, output_loading_info=True)

        missing_keys = loading_info["missing_keys"]
        unexpected_keys = loading_info["unexpected_keys"]
        self.assertIn("`TestModelBeta`", cl2.out)
        self.assertIn(warning_msg_beta, cl2.out)
        self.assertIn("beta_param", missing_keys)
        self.assertIn("bias_param", unexpected_keys)

    def test_isin_mps_friendly(self):
        """tests that our custom `isin_mps_friendly` matches `torch.isin`"""
        random_ids = torch.randint(0, 100, (100,))
        # We can match against an interger
        random_test_integer = torch.randint(0, 100, (1,)).item()
        self.assertTrue(
            torch.equal(
                torch.isin(random_ids, random_test_integer), isin_mps_friendly(random_ids, random_test_integer)
            )
        )
        # We can match against an 0D tensor
        random_test_tensor = torch.randint(0, 100, (1,)).squeeze()
        self.assertTrue(
            torch.equal(torch.isin(random_ids, random_test_tensor), isin_mps_friendly(random_ids, random_test_tensor))
        )
        # We can match against an 1D tensor (with many items)
        random_test_tensor = torch.randint(0, 100, (10,))
        self.assertTrue(
            torch.equal(torch.isin(random_ids, random_test_tensor), isin_mps_friendly(random_ids, random_test_tensor))
        )

    def test_can_generate(self):
        """Tests the behavior of `PreTrainedModel.can_generate` method."""
        logger = logging.get_logger("transformers.modeling_utils")
        logger.warning_once.cache_clear()

        # 1 - By default, a model CAN'T generate
        can_generate = BertModel.can_generate()
        self.assertFalse(can_generate)

        # 2 - The most common case for a model to be able to generate is to inherit from `GenerationMixin` directly
        class DummyBertWithMixin(BertModel, GenerationMixin):
            pass

        with CaptureLogger(logger) as cl:
            can_generate = DummyBertWithMixin.can_generate()
        self.assertTrue("" == cl.out)
        self.assertTrue(can_generate)

        # 3 - Alternatively, a model can implement a `generate` method
        class DummyBertWithGenerate(BertModel):
            def generate(self):
                pass

        with CaptureLogger(logger) as cl:
            can_generate = DummyBertWithGenerate.can_generate()
        self.assertTrue("" == cl.out)
        self.assertTrue(can_generate)

        # 4 - Finally, it can inherit from a model that can generate
        class DummyBertWithParent(DummyBertWithMixin):
            pass

        with CaptureLogger(logger) as cl:
            can_generate = DummyBertWithParent.can_generate()
        self.assertTrue("" == cl.out)
        self.assertTrue(can_generate)

        # 5 - BC: models with a custom `prepare_inputs_for_generation` can generate (it was assumed they inherited
        # `GenerationMixin`)
        class DummyBertWithPrepareInputs(BertModel):
            def prepare_inputs_for_generation(self):
                pass

        with CaptureLogger(logger) as cl:
            can_generate = DummyBertWithPrepareInputs.can_generate()
        self.assertTrue("it doesn't directly inherit from `GenerationMixin`" in cl.out)
        self.assertTrue(can_generate)

    def test_save_and_load_config_with_custom_generation(self):
        """
        Regression test for the ability to save and load a config with a custom generation kwarg (i.e. a parameter
        that gets moved to the generation config and reset on the model config)
        """
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)

        # The default for `num_beams` is 1 and `early_stopping` is False
        self.assertTrue(model.config.num_beams == 1)
        self.assertTrue(model.config.early_stopping is False)

        # When we save the model, this custom parameter should be moved to the generation config AND the model
        # config should contain `None`
        model.config.num_beams = 2
        model.config.early_stopping = True
        self.assertTrue(model.generation_config.num_beams == 1)  # unmodified generation config
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            new_model = T5ForConditionalGeneration.from_pretrained(tmp_dir)
            # moved to generation config
            self.assertTrue(new_model.generation_config.num_beams == 2)
            self.assertTrue(new_model.generation_config.early_stopping is True)
            # reset in the model config
            self.assertTrue(new_model.config.num_beams is None)
            self.assertTrue(new_model.config.early_stopping is None)

            # Sanity check: We can run `generate` with the new model without any warnings
            random_ids = torch.randint(0, 100, (1, 5))
            with warnings.catch_warnings(record=True) as w:
                new_model.generate(random_ids, max_new_tokens=3)
            self.assertTrue(len(w) == 0)

    def test_load_model_with_state_dict_only(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        state_dict = model.state_dict()
        config = model.config

        model_loaded = BertModel.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )
        self.assertTrue(check_models_equal(model, model_loaded))

    def test_load_model_with_state_dict_only_low_cpu_mem_usage(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        state_dict = model.state_dict()
        config = model.config

        model_loaded = BertModel.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict, low_cpu_mem_usage=True
        )
        self.assertTrue(check_models_equal(model, model_loaded))


@slow
@require_torch
class ModelOnTheFlyConversionTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.user = "huggingface-hub-ci"
        cls.token = os.getenv("HUGGINGFACE_PRODUCTION_USER_TOKEN", None)

        if cls.token is None:
            raise ValueError("Cannot run tests as secret isn't setup.")

        cls.api = HfApi(token=cls.token)

    def setUp(self) -> None:
        self.repo_name = f"{self.user}/test-model-on-the-fly-{uuid.uuid4()}"

    def tearDown(self) -> None:
        self.api.delete_repo(self.repo_name)

    def test_safetensors_on_the_fly_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_conversion_private(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, private=True)
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name, token=self.token)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_conversion_gated(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.put(
            f"https://huggingface.co/api/models/{self.repo_name}/settings", json={"gated": "auto"}, headers=headers
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, max_shard_size="200kb")
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion_private(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(
            self.repo_name, token=self.token, safe_serialization=False, max_shard_size="200kb", private=True
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion_gated(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, max_shard_size="200kb", safe_serialization=False)
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.put(
            f"https://huggingface.co/api/models/{self.repo_name}/settings", json={"gated": "auto"}, headers=headers
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    @unittest.skip(reason="Edge case, should work once the Space is updated`")
    def test_safetensors_on_the_fly_wrong_user_opened_pr(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, private=True)
        BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        # This should have opened a PR with the user's account
        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

        # We now switch the repo visibility to public
        self.api.update_repo_visibility(self.repo_name, private=False)

        # We once again call from_pretrained, which should call the bot to open a PR
        BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)

            bot_opened_pr = None
            bot_opened_pr_title = None

            for discussion in discussions:
                if discussion.author == "SFconvertbot":
                    bot_opened_pr = True
                    bot_opened_pr_title = discussion.title

            self.assertTrue(bot_opened_pr)
            self.assertEqual(bot_opened_pr_title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_specific_revision(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # Push a model on a given revision
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, revision="new-branch")

        # Try to convert the model on that revision should raise
        with self.assertRaises(EnvironmentError):
            BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token, revision="new-branch")

    def test_absence_of_safetensors_triggers_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # Download the model that doesn't have safetensors
        BertModel.from_pretrained(self.repo_name, token=self.token)

        for thread in threading.enumerate():
            if thread.name == "Thread-autoconversion":
                thread.join(timeout=10)

        discussions = self.api.get_repo_discussions(self.repo_name)

        bot_opened_pr = None
        bot_opened_pr_title = None

        for discussion in discussions:
            if discussion.author == "SFconvertbot":
                bot_opened_pr = True
                bot_opened_pr_title = discussion.title

        self.assertTrue(bot_opened_pr)
        self.assertEqual(bot_opened_pr_title, "Adding `safetensors` variant of this model")

    @mock.patch("transformers.safetensors_conversion.spawn_conversion")
    def test_absence_of_safetensors_triggers_conversion_failed(self, spawn_conversion_mock):
        spawn_conversion_mock.side_effect = HTTPError()

        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # The auto conversion is mocked to always raise; ensure that it doesn't raise in the main thread
        BertModel.from_pretrained(self.repo_name, token=self.token)


@require_torch
@is_staging_test
class ModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            model.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_with_description(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            COMMIT_DESCRIPTION = """
The commit description supports markdown synthax see:
```python
>>> form transformers import AutoConfig
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
```
"""
            commit_details = model.push_to_hub(
                tmp_repo.repo_id, use_auth_token=self._token, create_pr=True, commit_description=COMMIT_DESCRIPTION
            )
            self.assertEqual(commit_details.commit_description, COMMIT_DESCRIPTION)

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            model.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir, push_to_hub=True, token=self._token, repo_id=tmp_repo.repo_id)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_dynamic_model(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomConfig.register_for_auto_class()
            CustomModel.register_for_auto_class()

            config = CustomConfig(hidden_size=32)
            model = CustomModel(config)

            model.push_to_hub(tmp_repo.repo_id, token=self._token)
            # checks
            self.assertDictEqual(
                config.auto_map,
                {"AutoConfig": "custom_configuration.CustomConfig", "AutoModel": "custom_modeling.CustomModel"},
            )

            new_model = AutoModel.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            # Can't make an isinstance check because the new_model is from the CustomModel class of a dynamic module
            self.assertEqual(new_model.__class__.__name__, "CustomModel")
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

            config = AutoConfig.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            new_model = AutoModel.from_config(config, trust_remote_code=True)
            self.assertEqual(new_model.__class__.__name__, "CustomModel")

    def test_push_to_hub_with_tags(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            from huggingface_hub import ModelCard

            new_tags = ["tag-1", "tag-2"]

            CustomConfig.register_for_auto_class()
            CustomModel.register_for_auto_class()

            config = CustomConfig(hidden_size=32)
            model = CustomModel(config)

            self.assertTrue(model.model_tags is None)

            model.add_model_tags(new_tags)

            self.assertTrue(model.model_tags == new_tags)

            model.push_to_hub(tmp_repo.repo_id, token=self._token)

            loaded_model_card = ModelCard.load(tmp_repo.repo_id)
            self.assertEqual(loaded_model_card.data.tags, new_tags)


@require_torch
class AttentionMaskTester(unittest.TestCase):
    def check_non_causal(self, bsz, q_len, kv_len, mask_2d, mask_4d):
        mask_indices = (mask_2d != 1)[:, None].broadcast_to((bsz, q_len, kv_len))
        mask_4d_values = mask_4d[:, 0][mask_indices]
        is_inf = mask_4d_values == -float("inf")
        is_min = mask_4d_values == torch.finfo(mask_4d.dtype).min
        assert torch.logical_or(is_inf, is_min).all()

    def check_to_4d(self, mask_converter, q_len, kv_len, additional_mask=None, bsz=3):
        mask_2d = torch.ones((bsz, kv_len), device=torch_device, dtype=torch.long)

        if additional_mask is not None:
            for bsz_idx, seq_idx in additional_mask:
                mask_2d[bsz_idx, seq_idx] = 0

        mask_4d = mask_converter.to_4d(mask_2d, query_length=q_len, key_value_length=kv_len, dtype=torch.float32)

        assert mask_4d.shape == (bsz, 1, q_len, kv_len)

        # make sure there are no overflows
        assert mask_4d.min() != float("-inf")

        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)

            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
            if 0 in mask_2d:
                # at least causal mask + maybe more
                assert (mask_4d != 0).sum().cpu().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif not mask_converter.is_causal and context is None:
            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == 0
            if 0 in mask_2d:
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif mask_converter.is_causal and context is not None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = (q_len * (q_len - 1) // 2) + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked

            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
            if 0 in mask_2d:
                # at least causal mask + maybe more
                assert (mask_4d != 0).sum().cpu().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)

    def check_to_causal(self, mask_converter, q_len, kv_len, bsz=3):
        mask_4d = mask_converter.to_causal_4d(
            bsz, query_length=q_len, key_value_length=kv_len, device=torch_device, dtype=torch.float32
        )

        if q_len == 1 and mask_converter.sliding_window is None:
            # no causal mask if q_len is 1
            assert mask_4d is None
            return

        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)

            assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
        elif not mask_converter.is_causal and context is None:
            assert (mask_4d != 0).sum().cpu().item() == 0
        elif mask_converter.is_causal and context is not None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = (q_len * (q_len - 1) // 2) + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked

            assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked

    def compute_num_context_mask(self, kv_len, context, q_len):
        # This function computes the # of attention tokens that are added for
        # the sliding window
        c_mask_len = kv_len - context - 1
        num_mask_triangle = c_mask_len * (c_mask_len + 1) // 2
        cut_mask_len = max(c_mask_len - q_len, 0)
        num_cut_mask = cut_mask_len * (cut_mask_len + 1) // 2
        return num_mask_triangle - num_cut_mask

    def test_2d_to_4d_causal(self):
        mask_converter = AttentionMaskConverter(is_causal=True)

        # auto-regressive use case
        self.check_to_4d(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_4d(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=1, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=3, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

        # check that the mask does not overflow on causal masked tokens
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 0), (1, 0), (1, 1)])

    def test_2d_to_4d(self):
        mask_converter = AttentionMaskConverter(is_causal=False)

        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_2d_to_4d_causal_sliding(self):
        mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=5)

        # auto-regressive use case
        self.check_to_4d(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_4d(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=1, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=3, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_causal_mask(self):
        mask_converter = AttentionMaskConverter(is_causal=True)

        # auto-regressive use case
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)

    def test_causal_mask_sliding(self):
        mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=3)

        # auto-regressive use case
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)

    def test_torch_compile_fullgraph(self):
        model = Prepare4dCausalAttentionMaskModel()

        inputs_embeds = torch.rand([1, 3, 32])
        res_non_compiled = model(inputs_embeds)

        compiled_model = torch.compile(model, fullgraph=True)

        res_compiled = compiled_model(inputs_embeds)

        self.assertTrue(torch.equal(res_non_compiled, res_compiled))

        model = Create4dCausalAttentionMaskModel()

        inputs_embeds = torch.rand(2, 4, 16)
        res_non_compiled = model(inputs_embeds)

        compiled_model = torch.compile(model, fullgraph=True)
        res_compiled = compiled_model(inputs_embeds)

        self.assertTrue(torch.equal(res_non_compiled, res_compiled))

        model = Prepare4dAttentionMaskModel()

        mask = torch.ones(2, 4)
        mask[0, :2] = 0
        inputs_embeds = torch.rand(2, 4, 16)

        res_non_compiled = model(mask, inputs_embeds)

        compiled_model = torch.compile(model, fullgraph=True)
        res_compiled = compiled_model(mask, inputs_embeds)

        self.assertTrue(torch.equal(res_non_compiled, res_compiled))

    @require_torch
    @slow
    def test_unmask_unattended_left_padding(self):
        attention_mask = torch.Tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]]).to(torch.int64)

        expanded_mask = torch.Tensor(
            [
                [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[0, 0, 0], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(torch.int64)

        reference_output = torch.Tensor(
            [
                [[[1, 1, 1], [1, 1, 1], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[1, 1, 1], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(torch.int64)

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=1)

        self.assertTrue(torch.equal(result, reference_output))

        attention_mask = torch.Tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)
        min_inf = torch.finfo(torch.float32).min
        reference_output = torch.Tensor(
            [
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [min_inf, min_inf, 0, min_inf, min_inf],
                        [min_inf, min_inf, 0, 0, min_inf],
                        [min_inf, min_inf, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, min_inf, min_inf, min_inf, min_inf],
                        [0, 0, min_inf, min_inf, min_inf],
                        [0, 0, 0, min_inf, min_inf],
                        [0, 0, 0, 0, min_inf],
                        [0, 0, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 0, 0],
                        [min_inf, 0, min_inf, min_inf, min_inf],
                        [min_inf, 0, 0, min_inf, min_inf],
                        [min_inf, 0, 0, 0, min_inf],
                        [min_inf, 0, 0, 0, 0],
                    ]
                ],
            ]
        )

        self.assertTrue(torch.equal(reference_output, result))

    @require_torch
    @slow
    def test_unmask_unattended_right_padding(self):
        attention_mask = torch.Tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(torch.equal(expanded_mask, result))

    @require_torch
    @slow
    def test_unmask_unattended_random_mask(self):
        attention_mask = torch.Tensor([[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(torch.equal(expanded_mask, result))


@require_torch
class TestAttentionImplementation(unittest.TestCase):
    def test_error_no_sdpa_available(self):
        with self.assertRaises(ValueError) as cm:
            _ = AutoModel.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel", attn_implementation="sdpa")

        self.assertTrue(
            "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention"
            in str(cm.exception)
        )

        _ = AutoModel.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel")

    def test_error_no_flash_available(self):
        with self.assertRaises(ValueError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-tiny-model-private/tiny-random-MCTCTModel", attn_implementation="flash_attention_2"
            )

        self.assertTrue("does not support Flash Attention 2.0" in str(cm.exception))

    def test_error_no_flash_available_with_config(self):
        with self.assertRaises(ValueError) as cm:
            config = AutoConfig.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel")

            _ = AutoModel.from_pretrained(
                "hf-tiny-model-private/tiny-random-MCTCTModel", config=config, attn_implementation="flash_attention_2"
            )

        self.assertTrue("does not support Flash Attention 2.0" in str(cm.exception))

    def test_error_wrong_attn_implementation(self):
        with self.assertRaises(ValueError) as cm:
            _ = AutoModel.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel", attn_implementation="foo")

        self.assertTrue('The only possible arguments are `attn_implementation="eager"' in str(cm.exception))

    def test_not_available_flash(self):
        if is_flash_attn_2_available():
            self.skipTest(reason="Please uninstall flash-attn package to run test_not_available_flash")

        with self.assertRaises(ImportError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-internal-testing/tiny-random-GPTBigCodeModel", attn_implementation="flash_attention_2"
            )
        self.assertTrue("the package flash_attn seems to be not installed" in str(cm.exception))

    def test_not_available_flash_with_config(self):
        if is_flash_attn_2_available():
            self.skipTest(reason="Please uninstall flash-attn package to run test_not_available_flash")

        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-GPTBigCodeModel")

        with self.assertRaises(ImportError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-internal-testing/tiny-random-GPTBigCodeModel",
                config=config,
                attn_implementation="flash_attention_2",
            )

        self.assertTrue("the package flash_attn seems to be not installed" in str(cm.exception))

    def test_not_available_sdpa(self):
        if is_torch_sdpa_available():
            self.skipTest(reason="This test requires torch<=2.0")

        with self.assertRaises(ImportError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-internal-testing/tiny-random-GPTBigCodeModel", attn_implementation="sdpa"
            )

        self.assertTrue("PyTorch SDPA requirements in Transformers are not met" in str(cm.exception))


@require_torch
class TestTensorSharing(TestCasePlus):
    def test_disjoint(self):
        main = torch.zeros(10)
        a = main[:5]
        b = main[5:]
        state_dict = {"a": a, "b": b}

        shared_names, disjoint_names = _find_disjoint([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [])
        self.assertEqual(disjoint_names, ["a", "b"])

        a = main[::2]
        b = main[1::2]
        state_dict = {"a": a, "b": b}

        shared_names, disjoint_names = _find_disjoint([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [{"a", "b"}])
        self.assertEqual(disjoint_names, [])

    def test_identical(self):
        a = torch.zeros(10)
        b = a
        state_dict = {"a": a, "b": b}

        shared_names, identical_names = _find_identical([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [])
        self.assertEqual(identical_names, [{"a", "b"}])

        b = a[:5]
        state_dict = {"a": a, "b": b}

        shared_names, identical_names = _find_identical([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [{"a", "b"}])
        self.assertEqual(identical_names, [])
