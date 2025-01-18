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
import collections
import copy
import gc
import inspect
import math
import os
import os.path
import random
import re
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
from packaging import version
from parameterized import parameterized
from pytest import mark

import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
    is_torch_available,
    logging,
    set_seed,
)
from transformers.integrations import HfDeepSpeedConfig
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import (
    CaptureLogger,
    is_flaky,
    is_pt_flax_cross_test,
    is_pt_tf_cross_test,
    require_accelerate,
    require_bitsandbytes,
    require_deepspeed,
    require_flash_attn,
    require_non_xpu,
    require_safetensors,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_torch_multi_gpu,
    require_torch_sdpa,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    is_accelerate_available,
    is_flax_available,
    is_tf_available,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    is_torch_fx_available,
    is_torch_sdpa_available,
)
from transformers.utils.generic import ContextManagers, ModelOutput


if is_accelerate_available():
    from accelerate.utils import compute_module_sizes


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
    from torch import nn

    from transformers import MODEL_MAPPING, AdaptiveEmbedding
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.modeling_utils import load_state_dict, no_init_weights
    from transformers.pytorch_utils import id_tensor_storage


if is_tf_available():
    import tensorflow as tf

if is_flax_available():
    import jax.numpy as jnp

    from tests.utils.test_modeling_flax_utils import check_models_equal
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )

if is_torch_fx_available():
    from transformers.utils.fx import _FX_SUPPORTED_MODELS_WITH_KV_CACHE, symbolic_trace

if is_deepspeed_available():
    import deepspeed


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def _mock_init_weights(self, module):
    for name, param in module.named_parameters(recurse=False):
        # Use the first letter of the name to get a value and go from a <> -13 to z <> 12
        value = ord(name[0].lower()) - 110
        param.data.fill_(value)


def _mock_all_init_weights(self):
    # Prune heads if needed
    if self.config.pruned_heads:
        self.prune_heads(self.config.pruned_heads)

    import transformers.modeling_utils

    if transformers.modeling_utils._init_weights:
        for module in self.modules():
            module._is_hf_initialized = False
        # Initialize weights
        self.apply(self._initialize_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()


@contextmanager
def _deepspeed_zero3(ds_config):
    dschf = HfDeepSpeedConfig(ds_config)
    try:
        yield dschf
    finally:
        unset_hf_deepspeed_config()


def sdpa_kernel(enable_flash, enable_math, enable_mem_efficient):
    if version.parse(torch.__version__).release < version.parse("2.3").release:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient
        )

    backends = []
    if enable_flash:
        backends += [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    if enable_math:
        backends += [torch.nn.attention.SDPBackend.MATH]
    if enable_mem_efficient:
        backends += [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    return torch.nn.attention.sdpa_kernel(backends)


@require_torch
class ModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    fx_compatible = False
    test_torchscript = True
    test_pruning = True
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_head_masking = True
    test_mismatched_shapes = True
    test_missing_keys = True
    test_model_parallel = False
    # Used in `check_training_gradient_checkpointing` to NOT check all params having gradient (e.g. for some MOE models)
    test_all_params_have_gradient = True
    is_encoder_decoder = False
    has_attentions = True
    _is_composite = False
    model_split_percents = [0.5, 0.7, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
            inputs_dict = {
                k: v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                if isinstance(v, torch.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }
        elif model_class.__name__ in get_values(MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES):
            inputs_dict.pop("attention_mask")
        elif model_class.__name__ == MODEL_FOR_PRETRAINING_MAPPING_NAMES["hiera"]:
            config = self.model_tester.get_config()
            mask_spatial_shape = [
                i // s // ms for i, s, ms in zip(config.image_size, config.patch_stride, config.masked_unit_size)
            ]
            num_windows = math.prod(mask_spatial_shape)
            torch.manual_seed(0)
            inputs_dict["noise"] = torch.rand(self.model_tester.batch_size, num_windows)

        if return_labels:
            if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
                inputs_dict["labels"] = torch.ones(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
            ]:
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in get_values(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES):
                num_patches = self.model_tester.image_size // self.model_tester.patch_size
                inputs_dict["bool_masked_pos"] = torch.zeros(
                    (self.model_tester.batch_size, num_patches**2), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES):
                batch_size, num_channels, height, width = inputs_dict["pixel_values"].shape
                inputs_dict["labels"] = torch.zeros(
                    [self.model_tester.batch_size, height, width], device=torch_device
                ).long()

        return inputs_dict

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0
            out_2 = out_2[~np.isneginf(out_2)]

            out_1 = out1.cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            out_1 = out_1[~np.isneginf(out_1)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

                # Save and load second time because `from_pretrained` adds a bunch of new config fields
                # so we need to make sure those fields can be loaded back after saving
                # Simply init as `model(config)` doesn't add those fields
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_from_pretrained_no_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            state_dict = model.state_dict()

            new_model = model_class.from_pretrained(
                pretrained_model_name_or_path=None, config=config, state_dict=state_dict
            )
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_keep_in_fp32_modules(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if model_class._keep_in_fp32_modules is None:
                self.skipTest(reason="Model class has no _keep_in_fp32_modules attribute defined")

            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16)

                for name, param in model.named_parameters():
                    if any(n in model_class._keep_in_fp32_modules for n in name.split(".")):
                        self.assertTrue(param.dtype == torch.float32)
                    else:
                        self.assertTrue(param.dtype == torch.float16, name)

    def test_save_load_keys_to_ignore_on_save(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            _keys_to_ignore_on_save = getattr(model, "_keys_to_ignore_on_save", None)
            if _keys_to_ignore_on_save is None:
                continue

            # check the keys are in the original state_dict
            for k in _keys_to_ignore_on_save:
                self.assertIn(k, model.state_dict().keys(), "\n".join(model.state_dict().keys()))

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                output_model_file = os.path.join(tmpdirname, SAFE_WEIGHTS_NAME)
                state_dict_saved = safe_load_file(output_model_file)

                for k in _keys_to_ignore_on_save:
                    self.assertNotIn(k, state_dict_saved.keys(), "\n".join(state_dict_saved.keys()))

                # Test we can load the state dict in the model, necessary for the checkpointing API in Trainer.
                load_result = model.load_state_dict(state_dict_saved, strict=False)
                keys_to_ignore = set(model._keys_to_ignore_on_save)

                if hasattr(model, "_tied_weights_keys"):
                    keys_to_ignore.update(set(model._tied_weights_keys))

                self.assertTrue(len(load_result.missing_keys) == 0 or set(load_result.missing_keys) == keys_to_ignore)
                self.assertTrue(len(load_result.unexpected_keys) == 0)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(config)
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model.gradient_checkpointing_enable()
            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    def test_peft_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(config)
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model._hf_peft_config_loaded = True
            try:
                model.gradient_checkpointing_enable()
            except NotImplementedError:
                continue

            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    @is_flaky(description="low likelihood of failure, reason not yet discovered")
    def test_save_load_fast_init_from_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if config.__class__ not in MODEL_MAPPING:
            self.skipTest(reason=f"{config.__class__.__name__} not in MODEL_MAPPING")

        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(model_class):
                pass

            model_class_copy = CopyClass

            # make sure that all keys are expected for test
            model_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            model_class_copy._init_weights = _mock_init_weights
            model_class_copy.init_weights = _mock_all_init_weights

            model = base_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = model_class_copy.from_pretrained(tmpdirname)
                model_slow_init = model_class_copy.from_pretrained(tmpdirname, _fast_init=False)
                # Before we test anything

                for key in model_fast_init.state_dict().keys():
                    if isinstance(model_slow_init.state_dict()[key], torch.BoolTensor):
                        max_diff = (model_slow_init.state_dict()[key] ^ model_fast_init.state_dict()[key]).sum().item()
                    else:
                        max_diff = (model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    @slow
    @require_accelerate
    @mark.accelerate_tests
    def test_save_load_low_cpu_mem_usage(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as saved_model_path:
            for model_class in self.all_model_classes:
                model_to_save = model_class(config)
                model_to_save.save_pretrained(saved_model_path)

                self._check_save_load_low_cpu_mem_usage(model_class, saved_model_path)

    @slow
    @require_accelerate
    @mark.accelerate_tests
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as saved_model_path:
            for model_class in self.all_model_classes:
                model_to_save = model_class(config)
                model_to_save.config.save_pretrained(saved_model_path)
                torch.save(model_to_save.state_dict(), os.path.join(saved_model_path, "pytorch_model.bin"))

                self._check_save_load_low_cpu_mem_usage(model_class, saved_model_path)

    @slow
    @require_accelerate
    @mark.accelerate_tests
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        with tempfile.TemporaryDirectory() as saved_model_path:
            for model_class in self.all_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                model_to_save = model_class(config)

                model_to_save.save_pretrained(saved_model_path, safe_serialization=False)
                self._check_save_load_low_cpu_mem_usage(model_class, saved_model_path)

    def _check_save_load_low_cpu_mem_usage(self, model_class, saved_model_path):
        from accelerate.utils.modeling import named_module_tensors

        # Load the low usage and the normal models.
        model_low_usage, loading_info = model_class.from_pretrained(
            saved_model_path,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        model_non_low_usage = model_class.from_pretrained(saved_model_path)

        # Check that there were no missing keys.
        self.assertEqual(loading_info["missing_keys"], [])

        # The low_cpu_mem_usage=True causes the model params to be initialized with device=meta, and then
        # subsequently loaded with the correct values and onto the correct device. We check if there are any
        # remaining params that were not properly loaded.
        for name, tensor in named_module_tensors(model_low_usage, recurse=True):
            self.assertNotEqual(
                tensor.device,
                torch.device("meta"),
                "Tensor '" + name + "' has not been properly loaded and has device=meta.",
            )

        # Check that the parameters are equal.
        for p1, p2 in zip(model_low_usage.parameters(), model_non_low_usage.parameters()):
            self.assertEqual(p1.data.ne(p2.data).sum(), 0)

        # Check that the state dict keys are equal.
        self.assertEqual(set(model_low_usage.state_dict().keys()), set(model_non_low_usage.state_dict().keys()))

        # Check that the shared tensors are equal.
        tensor_ptrs1 = collections.defaultdict(list)
        for name, tensor in model_low_usage.state_dict().items():
            tensor_ptrs1[id_tensor_storage(tensor)].append(name)
        tied_params1 = [names for _, names in tensor_ptrs1.items() if len(names) > 1]

        tensor_ptrs2 = collections.defaultdict(list)
        for name, tensor in model_non_low_usage.state_dict().items():
            tensor_ptrs2[id_tensor_storage(tensor)].append(name)
        tied_params2 = [names for _, names in tensor_ptrs2.items() if len(names) > 1]

        self.assertEqual(tied_params1, tied_params2)

    def test_save_load_fast_init_to_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if config.__class__ not in MODEL_MAPPING:
            self.skipTest(reason=f"{config.__class__.__name__} not in MODEL_MAPPING")

        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(base_class):
                pass

            base_class_copy = CopyClass

            # make sure that all keys are expected for test
            base_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            base_class_copy._init_weights = _mock_init_weights
            base_class_copy.init_weights = _mock_all_init_weights

            model = model_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.config.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = base_class_copy.from_pretrained(tmpdirname)
                model_slow_init = base_class_copy.from_pretrained(tmpdirname, _fast_init=False)

                for key in model_fast_init.state_dict().keys():
                    if isinstance(model_slow_init.state_dict()[key], torch.BoolTensor):
                        max_diff = torch.max(
                            model_slow_init.state_dict()[key] ^ model_fast_init.state_dict()[key]
                        ).item()
                    else:
                        max_diff = torch.max(
                            torch.abs(model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key])
                        ).item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_torch_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if config.__class__ not in MODEL_MAPPING:
            self.skipTest(reason=f"{config.__class__.__name__} not in MODEL_MAPPING")

        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(base_class):
                pass

            base_class_copy = CopyClass

            # make sure that all keys are expected for test
            base_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            base_class_copy._init_weights = _mock_init_weights
            base_class_copy.init_weights = _mock_all_init_weights

            model = model_class(config)
            state_dict = model.state_dict()

            def check_equal(loaded):
                for key in state_dict.keys():
                    max_diff = torch.max(
                        state_dict()[key] ^ loaded[key]
                        if isinstance(state_dict[key], torch.BoolTensor)
                        else torch.abs(state_dict[key] - loaded[key])
                    ).item()
                    self.assertLessEqual(max_diff, 1e-6, msg=f"{key} not identical")

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pytorch_model.bin")
                torch.save(state_dict, pt_checkpoint_path, _use_new_zipfile_serialization=True)
                check_equal(load_state_dict(pt_checkpoint_path))
                torch.save(state_dict, pt_checkpoint_path, _use_new_zipfile_serialization=False)
                check_equal(load_state_dict(pt_checkpoint_path))

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            out_1 = out_1[~np.isneginf(out_1)]
            out_2 = out_2[~np.isneginf(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_batching_equivalence(self):
        """
        Tests that the model supports batching and that the output is the nearly the same for the same input in
        different batch sizes.
        (Why "nearly the same" not "exactly the same"? Batching uses different matmul shapes, which often leads to
        different results: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)
        """

        def get_tensor_equivalence_function(batched_input):
            # models operating on continuous spaces have higher abs difference than LMs
            # instead, we can rely on cos distance for image/speech models, similar to `diffusers`
            if "input_ids" not in batched_input:
                return lambda tensor1, tensor2: (
                    1.0 - F.cosine_similarity(tensor1.float().flatten(), tensor2.float().flatten(), dim=0, eps=1e-38)
                )
            return lambda tensor1, tensor2: torch.max(torch.abs(tensor1 - tensor2))

        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            elif isinstance(batched_object, dict):
                for batched_object_value, single_row_object_value in zip(
                    batched_object.values(), single_row_object.values()
                ):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            else:
                # indexing the first element does not always work
                # e.g. models that output similarity scores of size (N, M) would need to index [0, 0]
                slice_ids = [slice(0, index) for index in single_row_object.shape]
                batched_row = batched_object[slice_ids]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                self.assertTrue(
                    (equivalence(batched_row, single_row_object)) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={equivalence(batched_row, single_row_object)}."
                    ),
                )

        set_model_tester_for_less_flaky_test(self)

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)
        equivalence = get_tensor_equivalence_function(batched_input)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()
            set_model_for_less_flaky_test(model)

            batch_size = self.model_tester.batch_size
            single_row_input = {}
            for key, value in batched_input_prepared.items():
                if isinstance(value, torch.Tensor) and value.shape[0] % batch_size == 0:
                    # e.g. musicgen has inputs of size (bs*codebooks). in most cases value.shape[0] == batch_size
                    single_batch_shape = value.shape[0] // batch_size
                    single_row_input[key] = value[:single_batch_shape]
                else:
                    single_row_input[key] = value

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            if isinstance(model_batched_output, torch.Tensor):
                model_batched_output = {"model_output": model_batched_output}
                model_row_output = {"model_output": model_row_output}

            for key in model_batched_output:
                # DETR starts from zero-init queries to decoder, leading to cos_similarity = `nan`
                if hasattr(self, "zero_init_hidden_state") and "decoder_hidden_states" in key:
                    model_batched_output[key] = model_batched_output[key][1:]
                    model_row_output[key] = model_row_output[key][1:]
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if (
                    model_class.__name__
                    in [
                        *get_values(MODEL_MAPPING_NAMES),
                        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
                    ]
                    or not model_class.supports_gradient_checkpointing
                ):
                    # TODO (ydshieh): use `skipTest` once pytest-dev/pytest-subtests/pull/169 is merged
                    # self.skipTest(reason=f"`supports_gradient_checkpointing` is False for {model_class.__name__}.")
                    continue

                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                config.use_cache = False
                config.return_dict = True
                model = model_class(config)

                model.to(torch_device)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                model.train()

                # unfreeze additional layers
                for p in model.parameters():
                    p.requires_grad_(True)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                loss = model(**inputs).loss
                loss.backward()
                optimizer.step()

                if self.test_all_params_have_gradient:
                    for k, v in model.named_parameters():
                        if v.requires_grad:
                            self.assertTrue(v.grad is not None, f"{k} in {model_class.__name__} has no gradient!")

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
            ]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        # Scenario - 1 default behaviour
        self.check_training_gradient_checkpointing()

    def test_training_gradient_checkpointing_use_reentrant(self):
        # Scenario - 2 with `use_reentrant=True` - this is the default value that is used in pytorch's
        # torch.utils.checkpoint.checkpoint
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": True})

    def test_training_gradient_checkpointing_use_reentrant_false(self):
        # Scenario - 3 with `use_reentrant=False` pytorch suggests users to use this value for
        # future releases: https://pytorch.org/docs/stable/checkpoint.html
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class.__name__ in [
                    *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                    *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
                ]:
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    @slow
    def test_torchscript_simple(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torchscript(config, inputs_dict)

    @slow
    def test_torchscript_output_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_attentions = True
        self._create_and_check_torchscript(config, inputs_dict)

    @slow
    def test_torchscript_output_hidden_state(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        self._create_and_check_torchscript(config, inputs_dict)

    # This is copied from `torch/testing/_internal/jit_utils.py::clear_class_registry`
    def clear_torch_jit_class_registry(self):
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        # torch 1.8 has no `_clear_class_state` in `torch.jit._state`
        if hasattr(torch.jit._state, "_clear_class_state"):
            torch.jit._state._clear_class_state()

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to `False`")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            for attn_implementation in ["eager", "sdpa"]:
                if attn_implementation == "sdpa" and (not model_class._supports_sdpa or not is_torch_sdpa_available()):
                    continue

                configs_no_init._attn_implementation = attn_implementation
                model = model_class(config=configs_no_init)
                model.to(torch_device)
                model.eval()
                inputs = self._prepare_for_class(inputs_dict, model_class)

                main_input_name = model_class.main_input_name

                try:
                    if model.config.is_encoder_decoder:
                        model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                        main_input = inputs[main_input_name]
                        attention_mask = inputs["attention_mask"]
                        decoder_input_ids = inputs["decoder_input_ids"]
                        decoder_attention_mask = inputs["decoder_attention_mask"]
                        outputs = model(main_input, attention_mask, decoder_input_ids, decoder_attention_mask)
                        # `torchscript` doesn't work with outputs containing `Cache` object. However, #35235 makes
                        # several models to use `Cache` by default instead of the legacy cache (tuple), and
                        # their `torchscript` tests are failing. We won't support them anyway, but we still want to keep
                        # the tests for encoder models like `BERT`. So we skip the checks if the model's output contains
                        # a `Cache` object.
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (main_input, attention_mask, decoder_input_ids, decoder_attention_mask)
                        )
                    elif "bbox" in inputs and "image" in inputs:  # LayoutLMv2 requires additional inputs
                        input_ids = inputs["input_ids"]
                        bbox = inputs["bbox"]
                        image = inputs["image"].tensor
                        outputs = model(input_ids, bbox, image)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (input_ids, bbox, image), check_trace=False
                        )  # when traced model is checked, an error is produced due to name mangling
                    elif "bbox" in inputs:  # Bros requires additional inputs (bbox)
                        input_ids = inputs["input_ids"]
                        bbox = inputs["bbox"]
                        outputs = model(input_ids, bbox)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (input_ids, bbox), check_trace=False
                        )  # when traced model is checked, an error is produced due to name mangling
                    elif (
                        "pixel_values" in inputs and "prompt_pixel_values" in inputs and "prompt_masks" in inputs
                    ):  # SegGpt requires additional inputs
                        pixel_values = inputs["pixel_values"]
                        prompt_pixel_values = inputs["prompt_pixel_values"]
                        prompt_masks = inputs["prompt_masks"]
                        outputs = model(pixel_values, prompt_pixel_values, prompt_masks)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (pixel_values, prompt_pixel_values, prompt_masks), check_trace=False
                        )  # when traced model is checked, an error is produced due to name mangling
                    else:
                        main_input = inputs[main_input_name]

                        if model.config._attn_implementation == "sdpa":
                            trace_input = {main_input_name: main_input}

                            if "attention_mask" in inputs:
                                trace_input["attention_mask"] = inputs["attention_mask"]
                            else:
                                self.skipTest(reason="testing SDPA without attention_mask is not supported")

                            outputs = model(main_input, attention_mask=inputs["attention_mask"])
                            if any(isinstance(x, Cache) for x in outputs):
                                continue
                            # example_kwarg_inputs was introduced in torch==2.0, but it is fine here since SDPA has a requirement on torch>=2.1.
                            traced_model = torch.jit.trace(model, example_kwarg_inputs=trace_input)
                        else:
                            outputs = model(main_input)
                            if any(isinstance(x, Cache) for x in outputs):
                                continue
                            traced_model = torch.jit.trace(model, (main_input,))
                except RuntimeError:
                    self.fail("Couldn't trace module.")

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                    try:
                        torch.jit.save(traced_model, pt_file_name)
                    except Exception:
                        self.fail("Couldn't save module.")

                    try:
                        loaded_model = torch.jit.load(pt_file_name)
                    except Exception:
                        self.fail("Couldn't load module.")

                model.to(torch_device)
                model.eval()

                loaded_model.to(torch_device)
                loaded_model.eval()

                model_state_dict = model.state_dict()
                loaded_model_state_dict = loaded_model.state_dict()

                non_persistent_buffers = {}
                for key in loaded_model_state_dict.keys():
                    if key not in model_state_dict.keys():
                        non_persistent_buffers[key] = loaded_model_state_dict[key]

                loaded_model_state_dict = {
                    key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
                }

                self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

                model_buffers = list(model.buffers())
                for non_persistent_buffer in non_persistent_buffers.values():
                    found_buffer = False
                    for i, model_buffer in enumerate(model_buffers):
                        if torch.equal(non_persistent_buffer, model_buffer):
                            found_buffer = True
                            break

                    self.assertTrue(found_buffer)
                    model_buffers.pop(i)

                models_equal = True
                for layer_name, p1 in model_state_dict.items():
                    if layer_name in loaded_model_state_dict:
                        p2 = loaded_model_state_dict[layer_name]
                        if p1.data.ne(p2.data).sum() > 0:
                            models_equal = False

                self.assertTrue(models_equal)

                # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
                # (Even with this call, there are still memory leak by ~0.04MB)
                self.clear_torch_jit_class_registry()

    def test_torch_fx(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torch_fx_tracing(config, inputs_dict)

    def test_torch_fx_output_loss(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torch_fx_tracing(config, inputs_dict, output_loss=True)

    def _create_and_check_torch_fx_tracing(self, config, inputs_dict, output_loss=False):
        if not is_torch_fx_available() or not self.fx_compatible:
            self.skipTest(
                f"Either torch.fx is not available, or the model type {config.model_type} is not compatible with torch.fx"
            )

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.return_dict = False

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=output_loss)

            # We may want to test several inputs (various shapes, etc.).
            inputs_to_test = [inputs]

            if model.config.is_encoder_decoder:
                model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                labels = inputs.get("labels", None)
                input_names = [
                    "attention_mask",
                    "decoder_attention_mask",
                    "decoder_input_ids",
                    "input_features",
                    "input_ids",
                    "input_values",
                ]
                if labels is not None:
                    input_names.append("labels")
            else:
                input_names = [
                    "attention_mask",
                    "bbox",
                    "input_features",
                    "input_ids",
                    "input_values",
                    "inputs_embeds",
                    "pixel_values",
                    "token_type_ids",
                    "visual_feats",
                    "visual_pos",
                    "noise",
                ]

                labels = inputs.get("labels", None)
                start_positions = inputs.get("start_positions", None)
                end_positions = inputs.get("end_positions", None)
                if labels is not None:
                    input_names.append("labels")
                if start_positions is not None:
                    input_names.append("start_positions")
                if end_positions is not None:
                    input_names.append("end_positions")

                if model.config.model_type in _FX_SUPPORTED_MODELS_WITH_KV_CACHE:
                    input_names.append("past_key_values")

                    # Generally model_tester.prepare_config_and_inputs_for_common seem not to generate past key values inputs.
                    if "past_key_values" not in inputs:
                        batch_size = inputs[next(iter(inputs))].shape[0]
                        num_heads = model.config.num_attention_heads
                        head_dim = model.config.hidden_size // model.config.num_attention_heads

                        cache_shape = (batch_size, num_heads, 0, head_dim)
                        empty_pkv = tuple(
                            (
                                torch.rand(cache_shape, dtype=torch.float, device=torch_device),
                                torch.rand(cache_shape, dtype=torch.float, device=torch_device),
                            )
                            for i in range(model.config.num_hidden_layers)
                        )
                        empty_pkv = (
                            DynamicCache.from_legacy_cache(empty_pkv)
                            if model_class._supports_cache_class
                            else empty_pkv
                        )

                        cache_length = 9
                        cache_shape = (batch_size, num_heads, cache_length, head_dim)
                        non_empty_pkv = tuple(
                            (
                                torch.rand(cache_shape, dtype=torch.float, device=torch_device),
                                torch.rand(cache_shape, dtype=torch.float, device=torch_device),
                            )
                            for i in range(model.config.num_hidden_layers)
                        )
                        non_empty_pkv = (
                            DynamicCache.from_legacy_cache(non_empty_pkv)
                            if model_class._supports_cache_class
                            else non_empty_pkv
                        )

                        inps = copy.deepcopy(inputs_to_test[0])

                        inputs_to_test[0]["past_key_values"] = empty_pkv

                        inps["past_key_values"] = non_empty_pkv
                        inputs_to_test.append(inps)

                        past_mask = torch.ones(batch_size, cache_length, device=torch_device, dtype=torch.float)
                        inputs_to_test[1]["attention_mask"] = torch.cat(
                            (past_mask, inputs_to_test[1]["attention_mask"]), dim=1
                        )

                forward_parameters = inspect.signature(model.forward).parameters
                if "input_ids" in forward_parameters and "inputs_embeds" in forward_parameters:
                    inps = copy.deepcopy(inputs_to_test[0])

                    embedding_size = (
                        model.config.embedding_size
                        if getattr(model.config, "embedding_size", None) is not None
                        and model.config.model_type != "megatron-bert"
                        else model.config.hidden_size
                    )

                    if (
                        model.config.model_type in MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
                        and model.__class__.__name__
                        == MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES[model.config.model_type]
                    ):
                        batch_size, num_choices, sequence_length = inputs["input_ids"].shape
                        shape = (batch_size, num_choices, sequence_length, embedding_size)
                    elif inps["input_ids"].ndim == 2:
                        batch_size, sequence_length = inputs["input_ids"].shape
                        shape = (batch_size, sequence_length, embedding_size)
                    else:
                        self.skipTest("Unknown case")

                    del inps["input_ids"]
                    inps["inputs_embeds"] = torch.rand(shape, dtype=torch.float, device=torch_device)
                    inputs_to_test.append(inps)

            for inps in inputs_to_test:
                filtered_inputs = {k: v for (k, v) in inps.items() if k in input_names}
                input_names_to_trace = list(filtered_inputs.keys())

                if model.__class__.__name__ in set(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()) and (
                    not hasattr(model.config, "problem_type") or model.config.problem_type is None
                ):
                    model.config.problem_type = "single_label_classification"

                model.config.use_cache = "past_key_values" in input_names_to_trace

                traced_model = symbolic_trace(model, input_names_to_trace)

                with torch.no_grad():
                    traced_output = traced_model(**filtered_inputs)
                    model_output = model(**filtered_inputs)

                def flatten_output(output):
                    flatten = []
                    for x in output:
                        if isinstance(x, (tuple, list)):
                            flatten += flatten_output(x)
                        elif not isinstance(x, torch.Tensor):
                            continue
                        else:
                            flatten.append(x)
                    return flatten

                model_output = flatten_output(model_output)
                traced_output = flatten_output(traced_output)
                num_outputs = len(model_output)

                for i in range(num_outputs):
                    self.assertTrue(
                        torch.allclose(model_output[i], traced_output[i]),
                        f"traced {i}th output doesn't match model {i}th output for {model_class}",
                    )

                # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
                # (Even with this call, there are still memory leak by ~0.04MB)
                self.clear_torch_jit_class_registry()

    def test_headmasking(self):
        if not self.test_head_masking:
            self.skipTest(reason="Model does not support head masking")

        global_rng.seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        global_rng.seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            # Prepare head_mask
            # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior)
            head_mask = torch.ones(
                self.model_tester.num_hidden_layers,
                self.model_tester.num_attention_heads,
                device=torch_device,
            )
            head_mask[0, 0] = 0
            head_mask[-1, :-1] = 0
            head_mask.requires_grad_(requires_grad=True)
            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask
            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.forward)
                arg_names = [*signature.parameters.keys()]
                if "decoder_head_mask" in arg_names:  # necessary diferentiation because of T5 model
                    inputs["decoder_head_mask"] = head_mask
                if "cross_attn_head_mask" in arg_names:
                    inputs["cross_attn_head_mask"] = head_mask
            outputs = model(**inputs, return_dict=True)

            # Test that we can get a gradient back for importance score computation
            output = sum(t.sum() for t in outputs[0])
            output = output.sum()
            output.backward()
            multihead_outputs = head_mask.grad

            self.assertIsNotNone(multihead_outputs)
            self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)

            def check_attentions_validity(attentions):
                # Remove Nan
                for t in attentions:
                    self.assertLess(
                        torch.sum(torch.isnan(t)), t.numel() / 4
                    )  # Check we don't have more than 25% nans (arbitrary)
                attentions = [
                    t.masked_fill(torch.isnan(t), 0.0) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
                if len(attentions) > 2:  # encoder-decoder models have only 2 layers in each module
                    self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)

            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)

    def test_head_pruning(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            # TODO: To have this check, we will need at least 3 layers. Do we really need it?
            # self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_pretrained(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]
            self.assertEqual(attentions[0].shape[-3], 1)
            # TODO: To have this check, we will need at least 3 layers. Do we really need it?
            # self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_config_init(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            # TODO: To have this check, we will need at least 3 layers. Do we really need it?
            # self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_integration(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {1: [1, 2]}
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 0)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 0)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)

            heads_to_prune = {0: [0], 1: [1, 2]}
            model.prune_heads(heads_to_prune)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)

            self.assertDictEqual(model.config.pruned_heads, {0: [0], 1: [1, 2]})

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        if config.is_encoder_decoder:
            # Seq2Seq models
            encoder_hidden_states = outputs.encoder_hidden_states[0]
            encoder_hidden_states.retain_grad()

            decoder_hidden_states = outputs.decoder_hidden_states[0]
            decoder_hidden_states.retain_grad()

            if self.has_attentions:
                encoder_attentions = outputs.encoder_attentions[0]
                encoder_attentions.retain_grad()

                decoder_attentions = outputs.decoder_attentions[0]
                decoder_attentions.retain_grad()

                cross_attentions = outputs.cross_attentions[0]
                cross_attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(encoder_hidden_states.grad)
            self.assertIsNotNone(decoder_hidden_states.grad)

            if self.has_attentions:
                self.assertIsNotNone(encoder_attentions.grad)
                self.assertIsNotNone(decoder_attentions.grad)
                self.assertIsNotNone(cross_attentions.grad)
        else:
            # Encoder-/Decoder-only models
            hidden_states = outputs.hidden_states[0]
            hidden_states.retain_grad()

            if self.has_attentions:
                attentions = outputs.attentions[0]
                attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)

            if self.has_attentions:
                self.assertIsNotNone(attentions.grad)

    def test_feed_forward_chunking(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            torch.manual_seed(0)
            config.chunk_size_feed_forward = 1
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            self.assertTrue(torch.allclose(hidden_states_no_chunk, hidden_states_with_chunk, atol=1e-3))

    def test_resize_position_vector_embeddings(self):
        if not self.test_resize_position_embeddings:
            self.skipTest(reason="Model does not have position embeddings")

        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            max_position_embeddings = config.max_position_embeddings

            # Retrieve the embeddings and clone theme
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                encoder_cloned_embeddings = encoder_model_embed.weight.clone()
                decoder_cloned_embeddings = decoder_model_embed.weight.clone()
            else:
                model_embed = model.get_position_embeddings()
                cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the position embeddings with a larger max_position_embeddings increases
            # the model's postion embeddings size
            model.resize_position_embeddings(max_position_embeddings + 10)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings + 10)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] + 10)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] + 10)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the position embeddings with a smaller max_position_embeddings decreases
            # the model's max_position_embeddings
            model.resize_position_embeddings(max_position_embeddings - 5)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings - 5)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] - 5)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] - 5)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 5)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True

            if model.config.is_encoder_decoder:
                for p1, p2 in zip(encoder_cloned_embeddings, encoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
                for p1, p2 in zip(decoder_cloned_embeddings, decoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
            else:
                for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

    def test_resize_tokens_embeddings(self):
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to `False`")
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict.pop("labels", None)

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_embed_pre_resize = model.get_input_embeddings()
            type_model_embed_pre_resize = type(model_embed_pre_resize)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check to make sure the type of embeddings returned post resizing is same as type of input
            type_model_embed_post_resize = type(model_embed)
            self.assertEqual(type_model_embed_pre_resize, type_model_embed_post_resize)
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(model_embed.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, atol=1e-3, rtol=1e-1)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                model(**model_inputs)

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            # make sure that decoder_input_ids are resized as well
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                if "decoder_input_ids" in inputs_dict:
                    inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                model(**model_inputs)

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            del model
            del config
            # Copy again. config changed with embedding resizing (`vocab_size` changed)
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_vocab_size = config.get_text_config().vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertTrue(new_model_vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], new_model_vocab_size)
            self.assertTrue(new_model_vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

            # Test when `vocab_size` is smaller than `hidden_size`.
            del model
            del config
            # Copy again. config changed with embedding resizing (`vocab_size` changed)
            config = copy.deepcopy(original_config)
            config.vocab_size = 4
            config.pad_token_id = 3
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check to make sure the type of embeddings returned post resizing is same as type of input
            type_model_embed_post_resize = type(model_embed)
            self.assertEqual(type_model_embed_pre_resize, type_model_embed_post_resize)
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(model_embed.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, atol=1e-3, rtol=1e-1)

    @require_deepspeed
    @require_torch_accelerator
    def test_resize_tokens_embeddings_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_tokens_embeddings()

    @require_deepspeed
    @require_torch_multi_gpu
    def test_resize_tokens_embeddings_with_deepspeed_multi_gpu(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_tokens_embeddings()

    def test_resize_embeddings_untied(self):
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to `False`")

        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        original_config.tie_word_embeddings = False
        inputs_dict.pop("labels", None)

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untied embeddings")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.get_text_config().vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model(**self._prepare_for_class(inputs_dict, model_class))

            # Test multivariate resizing.
            model.resize_token_embeddings(model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(output_embeds.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(output_embeds.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(output_embeds.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(output_embeds.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(output_embeds.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, atol=1e-3, rtol=1e-1)
            # check if the old bias mean close to added bias mean.
            if output_embeds.bias is not None:
                if is_deepspeed_zero3_enabled():
                    with deepspeed.zero.GatheredParameters(output_embeds.bias, modifier_rank=None):
                        old_bias_mean = torch.mean(output_embeds.bias.data[:-10], axis=0)
                        new_bias_mean = torch.mean(output_embeds.bias.data[-10:], axis=0)
                else:
                    old_bias_mean = torch.mean(output_embeds.bias.data[:-10], axis=0)
                    new_bias_mean = torch.mean(output_embeds.bias.data[-10:], axis=0)

                torch.testing.assert_close(old_bias_mean, new_bias_mean, atol=1e-5, rtol=1e-2)

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model(**self._prepare_for_class(inputs_dict, model_class))

    @require_deepspeed
    @require_torch_accelerator
    def test_resize_embeddings_untied_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()

    @require_deepspeed
    @require_torch_multi_gpu
    def test_resize_embeddings_untied_with_deepspeed_multi_gpu(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()

    def test_model_get_set_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Embedding, AdaptiveEmbedding))

            new_input_embedding_layer = nn.Embedding(10, 10)
            model.set_input_embeddings(new_input_embedding_layer)
            self.assertEqual(model.get_input_embeddings(), new_input_embedding_layer)

            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            self.skipTest(reason="test_missing_keys is set to `False`")
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):
                extra_params = {k: v for k, v in model.named_parameters() if not k.startswith(base_model_prefix)}
                extra_params.update({k: v for k, v in model.named_buffers() if not k.startswith(base_model_prefix)})
                # Some models define this as None
                if model._keys_to_ignore_on_load_missing:
                    for key in model._keys_to_ignore_on_load_missing:
                        extra_params.pop(key, None)

                if not extra_params:
                    # In that case, we *are* on a head model, but every
                    # single key is not actual parameters and this is
                    # tested in `test_tied_model_weights_key_ignore` test.
                    continue

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)
                    self.assertGreater(len(loading_info["missing_keys"]), 0, model.__class__.__name__)

    def test_tie_model_weights(self):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to `False`")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            vocab_size = config.get_text_config().vocab_size
            model_tied.resize_token_embeddings(vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

    @require_safetensors
    def test_can_use_safetensors(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model_tied = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                try:
                    model_tied.save_pretrained(d, safe_serialization=True)
                except Exception as e:
                    raise Exception(f"Class {model_class.__name__} cannot be saved using safetensors: {e}")

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model_tied.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v, reloaded_state[k], msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}"
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

                # Checking the tensor sharing are correct
                ptrs = defaultdict(list)
                for k, v in model_tied.state_dict().items():
                    ptrs[v.data_ptr()].append(k)

                shared_ptrs = {k: v for k, v in ptrs.items() if len(v) > 1}

                for _, shared_names in shared_ptrs.items():
                    reloaded_ptrs = {reloaded_state[k].data_ptr() for k in shared_names}
                    self.assertEqual(
                        len(reloaded_ptrs),
                        1,
                        f"The shared pointers are incorrect, found different pointers for keys {shared_names}",
                    )

    def test_load_save_without_tied_weights(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.tie_word_embeddings = False
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v, reloaded_state[k], msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}"
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

    def test_tied_weights_keys(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.get_text_config().tie_word_embeddings = True
        for model_class in self.all_model_classes:
            model_tied = model_class(config)

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            tied_weight_keys = model_tied._tied_weights_keys if model_tied._tied_weights_keys is not None else []
            # Detect we get a hit for each key
            for key in tied_weight_keys:
                is_tied_key = any(re.search(key, p) for group in tied_params for p in group)
                self.assertTrue(is_tied_key, f"{key} is not a tied weight key for {model_class}.")

            # Removed tied weights found from tied params -> there should only be one left after
            for key in tied_weight_keys:
                for i in range(len(tied_params)):
                    tied_params[i] = [p for p in tied_params[i] if re.search(key, p) is None]

            tied_params = [group for group in tied_params if len(group) > 1]
            self.assertListEqual(
                tied_params,
                [],
                f"Missing `_tied_weights_keys` for {model_class}: add all of {tied_params} except one.",
            )

    def test_model_weights_reload_no_missing_tied_weights(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)

                # We are nuking ALL weights on file, so every parameter should
                # yell on load. We're going to detect if we yell too much, or too little.
                placeholder_dict = {"tensor": torch.tensor([1, 2])}
                safe_save_file(placeholder_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
                model_reloaded, infos = model_class.from_pretrained(tmp_dir, output_loading_info=True)

                prefix = f"{model_reloaded.base_model_prefix}."
                params = dict(model_reloaded.named_parameters())
                params.update(dict(model_reloaded.named_buffers()))
                param_names = {k[len(prefix) :] if k.startswith(prefix) else k for k in params.keys()}

                missing_keys = set(infos["missing_keys"])

                extra_missing = missing_keys - param_names
                # Remove tied weights from extra missing: they are normally not warned as missing if their tied
                # counterpart is present but here there are no weights at all so we do get the warning.
                ptrs = collections.defaultdict(list)
                for name, tensor in model_reloaded.state_dict().items():
                    ptrs[id_tensor_storage(tensor)].append(name)
                tied_params = [names for _, names in ptrs.items() if len(names) > 1]
                for group in tied_params:
                    group = {k[len(prefix) :] if k.startswith(prefix) else k for k in group}
                    # We remove the group from extra_missing if not all weights from group are in it
                    if len(group - extra_missing) > 0:
                        extra_missing = extra_missing - set(group)

                self.assertEqual(
                    extra_missing,
                    set(),
                    f"This model {model_class.__name__} might be missing some `keys_to_ignore`: {extra_missing}. "
                    f"For debugging, tied parameters are {tied_params}",
                )

                missed_missing = param_names - missing_keys
                # Remove nonpersistent buffers from missed_missing
                buffers = [n for n, _ in model_reloaded.named_buffers()]
                nonpersistent_buffers = {n for n in buffers if n not in model_reloaded.state_dict()}
                nonpersistent_buffers = {
                    k[len(prefix) :] if k.startswith(prefix) else k for k in nonpersistent_buffers
                }
                missed_missing = missed_missing - nonpersistent_buffers

                if model_reloaded._keys_to_ignore_on_load_missing is None:
                    expected_missing = set()
                else:
                    expected_missing = set(model_reloaded._keys_to_ignore_on_load_missing)
                self.assertEqual(
                    missed_missing,
                    expected_missing,
                    f"This model {model_class.__name__} ignores keys {missed_missing} but they look like real"
                    " parameters. If they are non persistent buffers make sure to instantiate them with"
                    " `persistent=False`",
                )

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    # model might return non-tensors objects (e.g. Cache class)
                    elif isinstance(tuple_object, torch.Tensor):
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _make_attention_mask_non_null(self, inputs_dict):
        """Make sure no sequence has all zeros as attention mask"""

        for k in ["attention_mask", "encoder_attention_mask", "decoder_attention_mask"]:
            if k in inputs_dict:
                attention_mask = inputs_dict[k]

                # Make sure no all 0s attention masks - to avoid failure at this moment.
                # Put `1` at the beginning of sequences to make it still work when combining causal attention masks.
                # TODO: remove this line once a fix regarding large negative values for attention mask is done.
                attention_mask = torch.cat(
                    [torch.ones_like(attention_mask[:, :1], dtype=attention_mask.dtype), attention_mask[:, 1:]], dim=-1
                )

                # Here we make the first sequence with all 0s as attention mask.
                # Currently, this will fail for `TFWav2Vec2Model`. This is caused by the different large negative
                # values, like `1e-4`, `1e-9`, `1e-30` and `-inf` for attention mask across models/frameworks.
                # TODO: enable this block once the large negative values thing is cleaned up.
                # (see https://github.com/huggingface/transformers/issues/14859)
                # attention_mask = torch.cat(
                #     [torch.zeros_like(attention_mask[:1], dtype=attention_mask.dtype), attention_mask[1:]],
                #     dim=0
                # )

                inputs_dict[k] = attention_mask

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _postprocessing_to_ignore_test_cases(self, tf_outputs, pt_outputs, model_class):
        """For temporarily ignoring some failed test cases (issues to be fixed)"""

        tf_keys = {k for k, v in tf_outputs.items() if v is not None}
        pt_keys = {k for k, v in pt_outputs.items() if v is not None}

        key_differences = tf_keys.symmetric_difference(pt_keys)

        if model_class.__name__ in [
            "FlaubertWithLMHeadModel",
            "FunnelForPreTraining",
            "ElectraForPreTraining",
            "XLMWithLMHeadModel",
        ]:
            for k in key_differences:
                if k in ["loss", "losses"]:
                    tf_keys.discard(k)
                    pt_keys.discard(k)
        elif model_class.__name__.startswith("GPT2"):
            # `TFGPT2` has `past_key_values` as a tensor while `GPT2` has it as a tuple.
            tf_keys.discard("past_key_values")
            pt_keys.discard("past_key_values")

        # create new outputs from the remaining fields
        new_tf_outputs = type(tf_outputs)(**{k: tf_outputs[k] for k in tf_keys})
        new_pt_outputs = type(pt_outputs)(**{k: pt_outputs[k] for k in pt_keys})

        return new_tf_outputs, new_pt_outputs

    # Copied from tests.test_modeling_tf_common.TFModelTesterMixin.check_pt_tf_outputs
    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=1e-4, name="outputs", attributes=None):
        """Check the outputs from PyTorch and TensorFlow models are close enough. Checks are done in a recursive way.

        Args:
            model_class: The class of the model that is currently testing. For example, `TFBertModel`,
                TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Mainly used for providing more informative
                error messages.
            name (`str`): The name of the output. For example, `output.hidden_states`, `output.attentions`, etc.
            attributes (`Tuple[str]`): The names of the output's element if the output is a tuple/list with each element
                being a named field in the output.
        """

        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f"{name}: The argument `attributes` should be a `tuple`")

        # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
        if isinstance(tf_outputs, ModelOutput):
            self.assertTrue(
                isinstance(pt_outputs, ModelOutput),
                f"{name}: `pt_outputs` should an instance of `ModelOutput` when `tf_outputs` is",
            )

            # Don't copy this block to model specific test file!
            # TODO: remove this method and this line after issues are fixed
            tf_outputs, pt_outputs = self._postprocessing_to_ignore_test_cases(tf_outputs, pt_outputs, model_class)

            tf_keys = [k for k, v in tf_outputs.items() if v is not None]
            pt_keys = [k for k, v in pt_outputs.items() if v is not None]

            self.assertEqual(tf_keys, pt_keys, f"{name}: Output keys differ between TF and PyTorch")

            # convert to the case of `tuple`
            # appending each key to the current (string) `name`
            attributes = tuple([f"{name}.{k}" for k in tf_keys])
            self.check_pt_tf_outputs(
                tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
            )

        # Allow `list` (e.g. `TransfoXLModelOutput.mems` is a list of tensors.)
        elif type(tf_outputs) in [tuple, list]:
            self.assertEqual(type(tf_outputs), type(pt_outputs), f"{name}: Output types differ between TF and PyTorch")
            self.assertEqual(len(tf_outputs), len(pt_outputs), f"{name}: Output lengths differ between TF and PyTorch")

            if attributes is not None:
                # case 1: each output has assigned name (e.g. a tuple form of a `ModelOutput`)
                self.assertEqual(
                    len(attributes),
                    len(tf_outputs),
                    f"{name}: The tuple `attributes` should have the same length as `tf_outputs`",
                )
            else:
                # case 2: each output has no assigned name (e.g. hidden states of each layer) -> add an index to `name`
                attributes = tuple([f"{name}_{idx}" for idx in range(len(tf_outputs))])

            for tf_output, pt_output, attr in zip(tf_outputs, pt_outputs, attributes):
                if isinstance(pt_output, DynamicCache):
                    pt_output = pt_output.to_legacy_cache()
                self.check_pt_tf_outputs(tf_output, pt_output, model_class, tol=tol, name=attr)

        elif isinstance(tf_outputs, tf.Tensor):
            self.assertTrue(
                isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `tf_outputs` is"
            )

            tf_outputs = tf_outputs.numpy()
            pt_outputs = pt_outputs.detach().to("cpu").numpy()

            self.assertEqual(
                tf_outputs.shape, pt_outputs.shape, f"{name}: Output shapes differ between TF and PyTorch"
            )

            # deal with NumPy's scalars to make replacing nan values by 0 work.
            if np.isscalar(tf_outputs):
                tf_outputs = np.array([tf_outputs])
                pt_outputs = np.array([pt_outputs])

            tf_nans = np.isnan(tf_outputs)
            pt_nans = np.isnan(pt_outputs)

            pt_outputs[tf_nans] = 0
            tf_outputs[tf_nans] = 0
            pt_outputs[pt_nans] = 0
            tf_outputs[pt_nans] = 0

            max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
            self.assertLessEqual(
                max_diff,
                tol,
                f"{name}: Difference between PyTorch and TF is {max_diff} (>= {tol}) for {model_class.__name__}",
            )
        else:
            raise ValueError(
                "`tf_outputs` should be an instance of `ModelOutput`, a `tuple`, or an instance of `tf.Tensor`. Got"
                f" {type(tf_outputs)} instead."
            )

    def prepare_tf_inputs_from_pt_inputs(self, pt_inputs_dict):
        tf_inputs_dict = {}
        for key, tensor in pt_inputs_dict.items():
            # skip key that does not exist in tf
            if isinstance(tensor, bool):
                tf_inputs_dict[key] = tensor
            elif key == "input_values":
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            elif key == "pixel_values":
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            elif key == "input_features":
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            # other general float inputs
            elif tensor.is_floating_point():
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)
            else:
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.int32)

        return tf_inputs_dict

    def check_pt_tf_models(self, tf_model, pt_model, pt_inputs_dict):
        tf_inputs_dict = self.prepare_tf_inputs_from_pt_inputs(pt_inputs_dict)

        # send pytorch inputs to the correct device
        pt_inputs_dict = {
            k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs_dict.items()
        }

        # send pytorch model to the correct device
        pt_model.to(torch_device)

        # Check predictions on first output (logits/hidden-states) are close enough given low-level computational differences
        pt_model.eval()

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs_dict)
        tf_outputs = tf_model(tf_inputs_dict)

        # tf models returned loss is usually a tensor rather than a scalar.
        # (see `hf_compute_loss`: it uses `tf.keras.losses.Reduction.NONE`)
        # Change it here to a scalar to match PyTorch models' loss
        tf_loss = getattr(tf_outputs, "loss", None)
        if tf_loss is not None:
            tf_outputs.loss = tf.math.reduce_mean(tf_loss)

        self.check_pt_tf_outputs(tf_outputs, pt_outputs, type(pt_model))

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self, allow_missing_keys=False):
        import transformers

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            tf_model_class_name = "TF" + model_class.__name__  # Add the "TF" at the beginning
            if not hasattr(transformers, tf_model_class_name):
                self.skipTest(reason="transformers does not have TF version of this model yet")

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
            # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
            # TODO: Use a uniform value for all models, make sure all tests pass without this processing, and remove it.
            self._make_attention_mask_non_null(inputs_dict)

            tf_model_class = getattr(transformers, tf_model_class_name)

            pt_model = model_class(config).eval()
            tf_model = tf_model_class(config)

            pt_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            pt_inputs_dict_with_labels = self._prepare_for_class(
                inputs_dict,
                model_class,
                # Not all models accept "labels" in the forward pass (yet :) )
                return_labels=True if "labels" in inspect.signature(model_class.forward).parameters.keys() else False,
            )

            # make sure only tf inputs are forward that actually exist in function args
            tf_input_keys = set(inspect.signature(tf_model.call).parameters.keys())

            # remove all head masks
            tf_input_keys.discard("head_mask")
            tf_input_keys.discard("cross_attn_head_mask")
            tf_input_keys.discard("decoder_head_mask")

            pt_inputs_dict = {k: v for k, v in pt_inputs_dict.items() if k in tf_input_keys}
            pt_inputs_dict_with_labels = {k: v for k, v in pt_inputs_dict_with_labels.items() if k in tf_input_keys}

            # For some models (e.g. base models), there is no label returned.
            # Set the input dict to `None` to avoid check outputs twice for the same input dicts.
            if not set(pt_inputs_dict_with_labels.keys()).symmetric_difference(pt_inputs_dict.keys()):
                pt_inputs_dict_with_labels = None

            # Check we can load pt model in tf and vice-versa with model => model functions
            # Here requires `tf_inputs_dict` to build `tf_model`
            tf_inputs_dict = self.prepare_tf_inputs_from_pt_inputs(pt_inputs_dict)
            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=tf_inputs_dict, allow_missing_keys=allow_missing_keys
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(
                pt_model, tf_model, allow_missing_keys=allow_missing_keys
            )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)
            # check with `labels`
            if pt_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict_with_labels)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(
                    tf_model, pt_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(
                    pt_model, tf_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)
            # check with `labels`
            if pt_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, pt_inputs_dict_with_labels)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    def check_pt_flax_outputs(self, fx_outputs, pt_outputs, model_class, tol=1e-4, name="outputs", attributes=None):
        """
        Args:
            model_class: The class of the model that is currently testing. For example, ..., etc.
            Currently unused, but it could make debugging easier and faster.

            names: A string, or a list of strings. These specify what fx_outputs/pt_outputs represent in the model outputs.
                Currently unused, but in the future, we could use this information to make the error message clearer
                by giving the name(s) of the output tensor(s) with large difference(s) between PT and Flax.
        """
        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f"{name}: The argument `attributes` should be a `tuple`")

        # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
        if isinstance(fx_outputs, ModelOutput):
            self.assertTrue(
                isinstance(pt_outputs, ModelOutput),
                f"{name}: `pt_outputs` should an instance of `ModelOutput` when `fx_outputs` is",
            )

            fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
            pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

            self.assertEqual(fx_keys, pt_keys, f"{name}: Output keys differ between Flax and PyTorch")

            # convert to the case of `tuple`
            # appending each key to the current (string) `name`
            attributes = tuple([f"{name}.{k}" for k in fx_keys])
            self.check_pt_flax_outputs(
                fx_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
            )

        # Allow `list` (e.g. `TransfoXLModelOutput.mems` is a list of tensors.)
        elif type(fx_outputs) in [tuple, list]:
            self.assertEqual(
                type(fx_outputs), type(pt_outputs), f"{name}: Output types differ between Flax and PyTorch"
            )
            self.assertEqual(
                len(fx_outputs), len(pt_outputs), f"{name}: Output lengths differ between Flax and PyTorch"
            )

            if attributes is not None:
                # case 1: each output has assigned name (e.g. a tuple form of a `ModelOutput`)
                self.assertEqual(
                    len(attributes),
                    len(fx_outputs),
                    f"{name}: The tuple `attributes` should have the same length as `fx_outputs`",
                )
            else:
                # case 2: each output has no assigned name (e.g. hidden states of each layer) -> add an index to `name`
                attributes = tuple([f"{name}_{idx}" for idx in range(len(fx_outputs))])

            for fx_output, pt_output, attr in zip(fx_outputs, pt_outputs, attributes):
                self.check_pt_flax_outputs(fx_output, pt_output, model_class, tol=tol, name=attr)

        elif isinstance(fx_outputs, jnp.ndarray):
            self.assertTrue(
                isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `fx_outputs` is"
            )

            # Using `np.asarray` gives `ValueError: assignment destination is read-only` at the line `fx_outputs[fx_nans] = 0`.
            fx_outputs = np.array(fx_outputs)
            pt_outputs = pt_outputs.detach().to("cpu").numpy()

            self.assertEqual(
                fx_outputs.shape, pt_outputs.shape, f"{name}: Output shapes differ between Flax and PyTorch"
            )

            # deal with NumPy's scalars to make replacing nan values by 0 work.
            if np.isscalar(fx_outputs):
                fx_outputs = np.array([fx_outputs])
                pt_outputs = np.array([pt_outputs])

            fx_nans = np.isnan(fx_outputs)
            pt_nans = np.isnan(pt_outputs)

            pt_outputs[fx_nans] = 0
            fx_outputs[fx_nans] = 0
            pt_outputs[pt_nans] = 0
            fx_outputs[pt_nans] = 0

            max_diff = np.amax(np.abs(fx_outputs - pt_outputs))
            self.assertLessEqual(
                max_diff, tol, f"{name}: Difference between PyTorch and Flax is {max_diff} (>= {tol})."
            )
        else:
            raise ValueError(
                "`fx_outputs` should be an instance of `ModelOutput`, a `tuple`, or an instance of `jnp.ndarray`. Got"
                f" {type(fx_outputs)} instead."
            )

    @is_pt_flax_cross_test
    def test_equivalence_pt_to_flax(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    self.skipTest(reason="No Flax model exists for this class")

                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                # load Flax class
                fx_model = fx_model_class(config, dtype=jnp.float32)

                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                # send pytorch inputs to the correct device
                pt_inputs = {
                    k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs.items()
                }

                # convert inputs to Flax
                fx_inputs = {k: np.array(v.to("cpu")) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = fx_model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs_loaded.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs_loaded, pt_outputs, model_class)

    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    self.skipTest(reason="No Flax model exists for this class")

                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                # load Flax class
                fx_model = fx_model_class(config, dtype=jnp.float32)

                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                # send pytorch inputs to the correct device
                pt_inputs = {
                    k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs.items()
                }

                # convert inputs to Flax
                fx_inputs = {k: np.array(v.to("cpu")) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = model_class.from_pretrained(
                        tmpdirname, from_flax=True, attn_implementation=fx_model.config._attn_implementation
                    )

                # send pytorch model to the correct device
                pt_model_loaded.to(torch_device)
                pt_model_loaded.eval()

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs_loaded.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs_loaded, model_class)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_MAPPING_NAMES):
                continue
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            model_forward_args = inspect.signature(model.forward).parameters
            if "inputs_embeds" not in model_forward_args:
                self.skipTest(reason="This model doesn't use `inputs_embeds`")

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            pad_token_id = config.pad_token_id if config.pad_token_id is not None else 1

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                # some models infer position ids/attn mask differently when input ids
                # by check if pad_token let's make sure no padding is in input ids
                not_pad_token_id = pad_token_id + 1 if max(0, pad_token_id - 1) == 0 else pad_token_id - 1
                input_ids[input_ids == pad_token_id] = not_pad_token_id
                del inputs["input_ids"]
                inputs_embeds = wte(input_ids)
                with torch.no_grad():
                    out_ids = model(input_ids=input_ids, **inputs)[0]
                    out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                encoder_input_ids[encoder_input_ids == pad_token_id] = max(0, pad_token_id + 1)
                decoder_input_ids[decoder_input_ids == pad_token_id] = max(0, pad_token_id + 1)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)
                inputs_embeds = wte(encoder_input_ids)
                decoder_inputs_embeds = wte(decoder_input_ids)
                with torch.no_grad():
                    out_ids = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, **inputs)[0]
                    out_embeds = model(
                        inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **inputs
                    )[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @require_non_xpu
    @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask", "decoder_head_mask", "cross_attn_head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    @require_torch_gpu
    @require_torch_multi_gpu
    def test_model_parallelization(self):
        if not self.test_model_parallel:
            self.skipTest(reason="test_model_parallel is set to False")

        # a candidate for testing_utils
        def get_current_gpu_memory_use():
            """returns a list of cuda memory allocations per GPU in MBs"""

            per_device_memory = []
            for id in range(torch.cuda.device_count()):
                with torch.cuda.device(id):
                    per_device_memory.append(torch.cuda.memory_allocated() >> 20)

            return per_device_memory

        # Needs a large model to see the difference.
        config = self.model_tester.get_large_model_config()

        for model_class in self.all_parallelizable_model_classes:
            torch.cuda.empty_cache()

            # 1. single gpu memory load + unload + memory measurements
            # Retrieve initial memory usage (can easily be ~0.6-1.5GB if cuda-kernels have been preloaded by previous tests)
            memory_at_start = get_current_gpu_memory_use()

            # Put model on device 0 and take a memory snapshot
            model = model_class(config)
            model.to("cuda:0")
            memory_after_model_load = get_current_gpu_memory_use()

            # The memory use on device 0 should be higher than it was initially.
            self.assertGreater(memory_after_model_load[0], memory_at_start[0])

            del model
            gc.collect()
            torch.cuda.empty_cache()

            # 2. MP test
            # it's essential to re-calibrate the usage before the next stage
            memory_at_start = get_current_gpu_memory_use()

            # Spread model layers over multiple devices
            model = model_class(config)
            model.parallelize()
            memory_after_parallelization = get_current_gpu_memory_use()

            # Assert that the memory use on all devices is higher than it was when loaded only on CPU
            for n in range(len(model.device_map.keys())):
                self.assertGreater(memory_after_parallelization[n], memory_at_start[n])

            # Assert that the memory use of device 0 is lower than it was when the entire model was loaded on it
            self.assertLess(memory_after_parallelization[0], memory_after_model_load[0])

            # Assert that the memory use of device 1 is higher than it was when the entire model was loaded
            # on device 0 and device 1 wasn't used at all
            self.assertGreater(memory_after_parallelization[1], memory_after_model_load[1])

            del model
            gc.collect()
            torch.cuda.empty_cache()

    @require_torch_gpu
    @require_torch_multi_gpu
    def test_model_parallel_equal_results(self):
        if not self.test_model_parallel:
            self.skipTest(reason="test_model_parallel is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_parallelizable_model_classes:
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)

            def cast_to_device(dictionary, device):
                output = {}
                for k, v in dictionary.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.to(device)
                    else:
                        output[k] = v

                return output

            model = model_class(config)
            output = model(**cast_to_device(inputs_dict, "cpu"))

            model.parallelize()

            parallel_output = model(**cast_to_device(inputs_dict, "cuda:0"))

            for value, parallel_value in zip(output, parallel_output):
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.allclose(value, parallel_value.to("cpu"), atol=1e-7))
                elif isinstance(value, (Tuple, List)):
                    for value_, parallel_value_ in zip(value, parallel_value):
                        self.assertTrue(torch.allclose(value_, parallel_value_.to("cpu"), atol=1e-7))

    def check_device_map_is_respected(self, model, device_map):
        for param_name, param in model.named_parameters():
            # Find device in device_map
            while len(param_name) > 0 and param_name not in device_map:
                param_name = ".".join(param_name.split(".")[:-1])
            if param_name not in device_map:
                raise ValueError("device map is incomplete, it does not contain any device for `param_name`.")

            param_device = device_map[param_name]
            if param_device in ["cpu", "disk"]:
                self.assertEqual(param.device, torch.device("meta"))
            elif param_device in ["mps"]:
                self.assertEqual(param.device, torch.device("mps"))
            else:
                # when loaded with device_map, `param_device` are integer values for cuda/xpu/npu/mlu
                self.assertEqual(param.device, torch.device(f"{torch_device}:{param_device}"))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_disk_offload_bin(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir, safe_serialization=False)

                with self.assertRaises(ValueError):
                    max_size = int(self.model_split_percents[0] * model_size)
                    max_memory = {0: max_size, "cpu": max_size}
                    # This errors out cause it's missing an offload folder
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}
                new_model = model_class.from_pretrained(
                    tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                    self.assertTrue(torch.allclose(a, b, atol=1e-5) for a, b in zip(base_output[0], new_output[0]))
                else:
                    self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_disk_offload_safetensors(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}

                # This doesn't error out as it's in safetensors and doesn't need an offload folder
                new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                    self.assertTrue(torch.allclose(a, b, atol=1e-5) for a, b in zip(base_output[0], new_output[0]))
                else:
                    self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_cpu_offload(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, "cpu": model_size * 2}
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, "cpu"})

                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                        self.assertTrue(torch.allclose(a, b, atol=1e-5) for a, b in zip(base_output[0], new_output[0]))
                    else:
                        self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_multi_accelerator
    def test_model_parallelism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, 1})
                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                        self.assertTrue(torch.allclose(a, b, atol=1e-5) for a, b in zip(base_output[0], new_output[0]))
                    else:
                        self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    def test_problem_types(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        problem_types = [
            {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
            {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
            {"title": "regression", "num_labels": 1, "dtype": torch.float},
        ]

        for model_class in self.all_model_classes:
            if model_class.__name__ not in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
            ]:
                continue

            for problem_type in problem_types:
                with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):
                    config.problem_type = problem_type["title"]
                    config.num_labels = problem_type["num_labels"]

                    model = model_class(config)
                    model.to(torch_device)
                    model.train()

                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                    if problem_type["num_labels"] > 1:
                        inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])

                    inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                    # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                    # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                    # they have the same size." which is a symptom something in wrong for the regression problem.
                    # See https://github.com/huggingface/transformers/issues/11780
                    with warnings.catch_warnings(record=True) as warning_list:
                        loss = model(**inputs).loss
                    for w in warning_list:
                        if "Using a target size that is different to the input size" in str(w.message):
                            raise ValueError(
                                f"Something is going wrong in the regression problem: intercepted {w.message}"
                            )

                    loss.backward()

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            self.skipTest(reason="test_missmatched_shapes is set to False")
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(RuntimeError):
                        new_model = AutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(RuntimeError):
                        new_model_without_prefix = AutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = AutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    new_model.to(torch_device)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    new_model_without_prefix.to(torch_device)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        if not self.test_mismatched_shapes:
            self.skipTest(reason="test_missmatched_shapes is set to False")
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)

        for model_class in self.all_model_classes:
            mappings = [
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
            ]
            is_classication_model = any(model_class.__name__ in get_values(mapping) for mapping in mappings)

            if not is_classication_model:
                continue

            # TODO: ydshieh
            is_special_classes = model_class.__name__ in [
                "wav2vec2.masked_spec_embed",
                "Wav2Vec2ForSequenceClassification",
                "CLIPForImageClassification",
                "RegNetForImageClassification",
                "ResNetForImageClassification",
                "UniSpeechSatForSequenceClassification",
                "Wav2Vec2BertForSequenceClassification",
                "PvtV2ForImageClassification",
                "Wav2Vec2ConformerForSequenceClassification",
                "WavLMForSequenceClassification",
                "SwiftFormerForImageClassification",
                "SEWForSequenceClassification",
                "BitForImageClassification",
                "SEWDForSequenceClassification",
                "SiglipForImageClassification",
                "HubertForSequenceClassification",
                "Swinv2ForImageClassification",
                "Data2VecAudioForSequenceClassification",
                "UniSpeechForSequenceClassification",
                "PvtForImageClassification",
                "ModernBertForSequenceClassification",
                "ModernBertForTokenClassification",
                "TimmWrapperForImageClassification",
            ]
            special_param_names = [
                r"^bit\.",
                r"^classifier\.weight",
                r"^classifier\.bias",
                r"^classifier\..+\.weight",
                r"^classifier\..+\.bias",
                r"^data2vec_audio\.",
                r"^dist_head\.",
                r"^head\.",
                r"^hubert\.",
                r"^pvt\.",
                r"^pvt_v2\.",
                r"^regnet\.",
                r"^resnet\.",
                r"^sew\.",
                r"^sew_d\.",
                r"^swiftformer\.",
                r"^swinv2\.",
                r"^transformers\.models\.swiftformer\.",
                r"^timm_model\.",
                r"^unispeech\.",
                r"^unispeech_sat\.",
                r"^vision_model\.",
                r"^wav2vec2\.",
                r"^wav2vec2_bert\.",
                r"^wav2vec2_conformer\.",
                r"^wavlm\.",
            ]

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(configs_no_init)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(RuntimeError):
                        new_model = model_class.from_pretrained(tmp_dir, num_labels=42)

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = model_class.from_pretrained(tmp_dir, num_labels=42, ignore_mismatched_sizes=True)
                    self.assertIn("the shapes did not match", cl.out)

                    for name, param in new_model.named_parameters():
                        if param.requires_grad:
                            param_mean = ((param.data.mean() * 1e9).round() / 1e9).item()
                            if not (
                                is_special_classes
                                and any(len(re.findall(target, name)) > 0 for target in special_param_names)
                            ):
                                self.assertIn(
                                    param_mean,
                                    [0.0, 1.0],
                                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                                )
                            else:
                                # Here we allow the parameters' mean to be in the range [-5.0, 5.0] instead of being
                                # either `0.0` or `1.0`, because their initializations are not using
                                # `config.initializer_factor` (or something similar). The purpose of this test is simply
                                # to make sure they are properly initialized (to avoid very large value or even `nan`).
                                self.assertGreaterEqual(
                                    param_mean,
                                    -5.0,
                                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                                )
                                self.assertLessEqual(
                                    param_mean,
                                    5.0,
                                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                                )

    def test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist(self):
        # 1. Create a dummy class. Should have buffers as well? To make sure we test __init__
        class MyClass(PreTrainedModel):
            config_class = PretrainedConfig

            def __init__(self, config=None):
                super().__init__(config if config is not None else PretrainedConfig())
                self.linear = nn.Linear(10, config.num_labels, bias=True)
                self.embedding = nn.Embedding(10, 10)
                self.std = 1

            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    module.weight.data = nn.init.kaiming_uniform_(module.weight.data, np.sqrt(5))
                    if module.bias is not None:
                        module.bias.data = module.bias.data.normal_(mean=0.0, std=self.std)

        # Used to make sure the weights with matched shape are loaded correctly
        config = PretrainedConfig()
        config.num_labels = 3
        model = MyClass(config=config)

        # Used to make sure the weights with mismatched shape are properly initialized
        set_seed(0)
        config = PretrainedConfig()
        config.num_labels = 4
        # not to init. the weights during the creation: to match the logic in `from_pretrained`, so we can keep the
        # same sequence of random ops in the execution path to allow us to compare `target_model` and `new_model` below
        # for `linear` part.
        with ContextManagers([no_init_weights(True)]):
            target_model = MyClass(config=config)
        target_model.apply(target_model._initialize_weights)

        with tempfile.TemporaryDirectory() as tmpdirname:
            state_dict = model.state_dict()
            del state_dict["linear.weight"]

            model.config.save_pretrained(tmpdirname)
            torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

            set_seed(0)
            new_model = MyClass.from_pretrained(tmpdirname, num_labels=4, ignore_mismatched_sizes=True)

            for key in new_model.state_dict().keys():
                # check weight values for weights with matched shapes are identical
                # (i.e. correctly loaded from the checkpoint)
                if key not in ["linear.weight", "linear.bias"]:
                    max_diff = torch.max(torch.abs(model.state_dict()[key] - new_model.state_dict()[key]))
                    self.assertLessEqual(
                        max_diff.item(),
                        1e-6,
                        msg=f"the weight values for `{key}` in `new_model` and `model` are  not identical",
                    )
                else:
                    # check we have some mismatched shapes
                    self.assertNotEqual(
                        model.state_dict()[key].shape,
                        new_model.state_dict()[key].shape,
                        msg=f"the weight shapes for {key} in `model` and `new_model` should differ",
                    )
                    # check the weights with mismatched shape are properly initialized
                    max_diff = torch.max(torch.abs(new_model.state_dict()[key] - target_model.state_dict()[key]))
                    self.assertLessEqual(
                        max_diff.item(),
                        1e-6,
                        msg=f"the weight values for `{key}` in `new_model` and `target_model` are not identical",
                    )

    def test_model_is_small(self):
        # Just a consistency check to make sure we are not running tests on 80M parameter models.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            num_params = model.num_parameters()
            assert (
                num_params < 1000000
            ), f"{model_class} is too big for the common tests ({num_params})! It should have 1M max."

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    dummy_attention_mask = dummy_attention_mask[:1]
                    dummy_attention_mask[:, 1:] = 1
                    dummy_attention_mask[:, :1] = 0

                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[:1]

                    outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                else:
                    outputs = model(dummy_input, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                if model.config.is_encoder_decoder:
                    other_inputs = {
                        "decoder_input_ids": decoder_input_ids,
                        "decoder_attention_mask": dummy_attention_mask,
                        "output_hidden_states": True,
                    }
                    if dummy_attention_mask is not None:
                        other_inputs["attention_mask"] = dummy_attention_mask

                    outputs = model(dummy_input, **other_inputs)
                    outputs_fa = model_fa(dummy_input, **other_inputs)
                else:
                    other_inputs = {
                        "output_hidden_states": True,
                    }
                    if dummy_attention_mask is not None:
                        other_inputs["attention_mask"] = dummy_attention_mask

                    outputs = model(dummy_input, **other_inputs)
                    outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[1:], logits[1:], atol=4e-2, rtol=4e-2)

                # check with inference + dropout
                model.train()
                _ = model_fa(dummy_input, **other_inputs)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    dummy_attention_mask = dummy_attention_mask[:1]
                    dummy_attention_mask[:, :-1] = 1
                    dummy_attention_mask[:, -1:] = 0

                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[:1]

                    outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                else:
                    outputs = model(dummy_input, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                if model.config.is_encoder_decoder:
                    other_inputs = {
                        "decoder_input_ids": decoder_input_ids,
                        "decoder_attention_mask": dummy_attention_mask,
                        "output_hidden_states": True,
                    }
                    if dummy_attention_mask is not None:
                        other_inputs["attention_mask"] = dummy_attention_mask

                    outputs = model(dummy_input, **other_inputs)
                    outputs_fa = model_fa(dummy_input, **other_inputs)
                else:
                    other_inputs = {
                        "output_hidden_states": True,
                    }
                    if dummy_attention_mask is not None:
                        other_inputs["attention_mask"] = dummy_attention_mask

                    outputs = model(dummy_input, **other_inputs)
                    outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[:-1], logits[:-1], atol=4e-2, rtol=4e-2)

    def test_attn_implementation_composite_models(self):
        """
        Tests if composite models can receive a dict object as attn_implementation, where each key should be
        one of the sub-configs from the model's config.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if not self._is_composite:
                self.skipTest("Model is not a composite model.")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # set eager as it will be the one supported in all models
            # we just need to test if passing 'attn_implementation' as a dict fails or not
            attn_implementation_per_subconfig = {}
            for key in config.sub_configs.keys():
                attn_implementation_per_subconfig[key] = "eager"

            config._attn_implementation = attn_implementation_per_subconfig
            model = model_class(config)
            for key in config.sub_configs.keys():
                sub_config = getattr(model.config, key)
                self.assertTrue(sub_config._attn_implementation == "eager")

            for name, submodule in model.named_modules():
                class_name = submodule.__class__.__name__
                if (
                    "SdpaAttention" in class_name
                    or "SdpaSelfAttention" in class_name
                    or "FlashAttention" in class_name
                ):
                    raise ValueError(f"The eager model should not have SDPA/FA2 attention layers but got {class_name}")

    @require_torch_sdpa
    def test_sdpa_can_dispatch_non_composite_models(self):
        """
        Tests if non-composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa or self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError(f"The eager model should not have SDPA attention layers but got {class_name}")

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                vision_model_names = {"visual", "image_tower", "vision_tower", "vision_model"}
                language_model_names = {"language_model", "model", "text_model"}
                vision_model_name = [name for name in vision_model_names if hasattr(model_sdpa, name)][0]
                language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)][0]

                vision_model_sdpa = getattr(model, vision_model_name)
                language_model_sdpa = getattr(model, language_model_name)
                text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
                vision_attn = "sdpa" if vision_model_sdpa._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
                self.assertTrue(vision_model_sdpa.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(getattr(model_eager, language_model_name).config._attn_implementation == "eager")
                self.assertTrue(getattr(model_eager, vision_model_name).config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            model = model_class(config)
            # FIXME: we deactivate boolean mask for models using "use_mask_token" in their constructors.
            # These models support masking only in the case `use_mask_token=True`. Otherwise they cannot consume an input mask.
            # This means that the class needs to be instantiated much later, after `use_mask` is set, which means a significant refactor of the code.
            # However masking there is not done at any layers that matters (i.e self-attention), therefore we can safely deactivate it.
            deactivate_mask = "use_mask_token" in inspect.signature(model_class).parameters
            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                try:
                    model_sdpa = model_class.from_pretrained(
                        tmpdirname, torch_dtype=torch_dtype, attn_implementation="sdpa"
                    )
                except ValueError:
                    model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device, dtype=torch_dtype)

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device, dtype=torch_dtype)

                set_model_for_less_flaky_test(model_eager)
                set_model_for_less_flaky_test(model_sdpa)

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 16 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for output_attentions in [True, False]:
                            can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                            if not (self.has_attentions and can_output_attn) and output_attentions:
                                continue
                            # TODO: if we can also check with `batch_size=1` without being flaky?
                            for batch_size in [7]:
                                dummy_input = inputs_dict[model.main_input_name]

                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    dummy_input = dummy_input.to(torch_dtype)

                                dummy_input = dummy_input[:batch_size]
                                if dummy_input.shape[0] != batch_size:
                                    if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                        extension = torch.rand(
                                            batch_size - dummy_input.shape[0],
                                            *dummy_input.shape[1:],
                                            dtype=torch_dtype,
                                            device=torch_device,
                                        )
                                        dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                    else:
                                        extension = torch.randint(
                                            high=5,
                                            size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                            dtype=dummy_input.dtype,
                                            device=torch_device,
                                        )
                                        dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                                if not use_mask:
                                    dummy_attention_mask = None
                                else:
                                    dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                    if dummy_attention_mask is None:
                                        if is_encoder_decoder:
                                            seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                        else:
                                            seqlen = dummy_input.shape[-1]
                                        dummy_attention_mask = (
                                            torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                        )

                                    dummy_attention_mask = dummy_attention_mask[:batch_size]
                                    if dummy_attention_mask.shape[0] != batch_size:
                                        extension = torch.ones(
                                            batch_size - dummy_attention_mask.shape[0],
                                            *dummy_attention_mask.shape[1:],
                                            dtype=dummy_attention_mask.dtype,
                                            device=torch_device,
                                        )
                                        dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                        dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                    dummy_attention_mask[:] = 1
                                    if padding_side == "left":
                                        dummy_attention_mask[-1, :2] = 0
                                        dummy_attention_mask[-1, 2:] = 1
                                    elif padding_side == "right":
                                        dummy_attention_mask[-1, -2:] = 0
                                        dummy_attention_mask[-1, :-2] = 1

                                for enable_kernels in [False, True]:
                                    failcase = f"padding_side={padding_side}, use_mask={use_mask}, enable_kernels={enable_kernels}"
                                    if is_encoder_decoder:
                                        decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[
                                            :batch_size
                                        ]
                                        if decoder_input_ids.shape[0] != batch_size:
                                            extension = torch.ones(
                                                batch_size - decoder_input_ids.shape[0],
                                                *decoder_input_ids.shape[1:],
                                                dtype=decoder_input_ids.dtype,
                                                device=torch_device,
                                            )
                                            decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                                            decoder_input_ids = decoder_input_ids.to(torch_device)

                                        # TODO: never an `attention_mask` arg here?
                                        processed_inputs = {
                                            model.main_input_name: dummy_input,
                                            "decoder_input_ids": decoder_input_ids,
                                            "decoder_attention_mask": dummy_attention_mask,
                                            "output_hidden_states": True,
                                        }
                                    else:
                                        processed_inputs = {
                                            model.main_input_name: dummy_input,
                                            "output_hidden_states": True,
                                        }

                                        # Otherwise fails for e.g. WhisperEncoderModel
                                        if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                                            processed_inputs["attention_mask"] = dummy_attention_mask

                                        if (
                                            self.has_attentions
                                            and "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                                        ):
                                            processed_inputs["output_attentions"] = output_attentions
                                    if not deactivate_mask and (
                                        "bool_masked_pos" in inspect.signature(model_eager.forward).parameters
                                    ):
                                        dummy_mask = torch.ones((self.model_tester.num_masks,))

                                        # In case of additional token (like class) we define a custom `mask_length`
                                        if hasattr(self.model_tester, "mask_length"):
                                            mask_length = self.model_tester.mask_length - dummy_mask.size(0)
                                        else:
                                            mask_length = self.model_tester.seq_length - dummy_mask.size(0)
                                        dummy_mask = torch.cat([dummy_mask, torch.zeros(mask_length)])
                                        dummy_bool_masked_pos = dummy_mask.expand(batch_size, -1).bool()
                                        processed_inputs["bool_masked_pos"] = dummy_bool_masked_pos.to(torch_device)

                                    if "noise" in inspect.signature(model_eager.forward).parameters:
                                        np.random.seed(2)
                                        num_patches = int(
                                            (self.model_tester.image_size // self.model_tester.patch_size) ** 2
                                        )
                                        noise = np.random.uniform(size=(batch_size, num_patches))
                                        processed_inputs["noise"] = torch.from_numpy(noise)

                                    # TODO: test gradients as well (& for FA2 as well!)
                                    with torch.no_grad():
                                        with sdpa_kernel(
                                            enable_flash=enable_kernels,
                                            enable_math=True,
                                            enable_mem_efficient=enable_kernels,
                                        ):
                                            prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                                            outputs_eager = model_eager(**prepared_inputs)
                                            outputs_sdpa = model_sdpa(**prepared_inputs)

                                    if hasattr(outputs_eager, "vision_hidden_states"):
                                        logits_eager = outputs_eager.vision_hidden_states[-1]
                                        logits_sdpa = outputs_sdpa.vision_hidden_states[-1]
                                    else:
                                        logits_eager = (
                                            outputs_eager.hidden_states[-1]
                                            if not is_encoder_decoder
                                            else outputs_eager.decoder_hidden_states[-1]
                                        )
                                        logits_sdpa = (
                                            outputs_sdpa.hidden_states[-1]
                                            if not is_encoder_decoder
                                            else outputs_sdpa.decoder_hidden_states[-1]
                                        )

                                    if torch_device in ["cpu", "cuda"]:
                                        atol = atols[torch_device, enable_kernels, torch_dtype]
                                        rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                    elif torch_device == "xpu":
                                        # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                                        # which is implemented on PyTorch level using aten operators and is
                                        # device agnostic with respect to implementation of each aten operator.
                                        atol = atols["cuda", False, torch_dtype]
                                        rtol = rtols["cuda", False, torch_dtype]
                                    else:
                                        atol = 1e-7
                                        rtol = 1e-4

                                    # Masked tokens output slightly deviates - we don't mind that.
                                    if use_mask:
                                        _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                                        _logits_eager = torch.zeros_like(input=logits_eager)

                                        _logits_sdpa[:-1] = logits_sdpa[:-1]
                                        _logits_eager[:-1] = logits_eager[:-1]

                                        if padding_side == "left":
                                            _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                                            _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                                        elif padding_side == "right":
                                            _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                                            _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                                        logits_sdpa = _logits_sdpa
                                        logits_eager = _logits_eager

                                    results = [
                                        torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                                        for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
                                    ]
                                    # If 80% batch elements have matched results, it's fine
                                    if np.mean(results) < 0.8:
                                        fail_cases.append(
                                            get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                        )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    @require_torch_sdpa
    @require_torch_gpu
    @slow
    def test_sdpa_can_dispatch_on_flash(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        torch.compiler.reset()
        compute_capability = torch.cuda.get_device_capability()
        major, _ = compute_capability

        if not torch.version.cuda or major < 8:
            self.skipTest(reason="This test requires an NVIDIA GPU with compute capability >= 8.0")

        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            if config.model_type in ["llava", "llava_next", "vipllava", "video_llava"]:
                self.skipTest(
                    reason="Llava-like models currently (transformers==4.39.1) requires an attention_mask input"
                )
            if config.model_type in ["paligemma"]:
                self.skipTest(
                    "PaliGemma-like models currently (transformers==4.41.0) requires an attention_mask input"
                )
            if config.model_type in ["idefics", "idefics2", "idefics3"]:
                self.skipTest(reason="Idefics currently (transformers==4.39.1) requires an image_attention_mask input")
            if config.model_type in ["sam"]:
                self.skipTest(reason="SAM requires an attention_mask input for relative positional embeddings")
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa")
                model.to(torch_device)

                inputs_dict.pop("attention_mask", None)
                inputs_dict.pop("decoder_attention_mask", None)

                for name, inp in inputs_dict.items():
                    if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float32, torch.float16]:
                        inputs_dict[name] = inp.to(torch.float16)

                with sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    _ = model(**inputs_dict)

    @require_non_xpu
    @require_torch_sdpa
    @require_torch_accelerator
    @slow
    def test_sdpa_can_compile_dynamic(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")
        torch.compiler.reset()
        if "cuda" in torch_device:
            compute_capability = torch.cuda.get_device_capability()
            major, _ = compute_capability

            if not torch.version.cuda or major < 8:
                self.skipTest(reason="This test requires an NVIDIA GPU with compute capability >= 8.0")

        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            if config.model_type in ["dbrx"]:
                self.skipTest(
                    "DBRX (transformers==4.40) requires a modification to support dynamic shapes with compile."
                )
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa")
                model.to(torch_device)

                # For PyTorch 2.1 - 2.3.0 set `dynamic=True`. In the future setting `dynamic=None` and using `torch._dynamo.mark_dynamic()`
                # on input tensors will be required. `mark_dynamic` currently raises inconsistent shape errors.
                model = torch.compile(model, dynamic=True)

                inputs_dict.pop("attention_mask", None)
                inputs_dict.pop("decoder_attention_mask", None)
                for name, inp in inputs_dict.items():
                    if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float32, torch.float16]:
                        inputs_dict[name] = inp.to(torch.float16)

                # use no_grad to save some memory
                with torch.no_grad():
                    _ = model(**inputs_dict)

    @require_torch_sdpa
    def test_sdpa_matches_eager_sliding_window(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        WINDOW_ATTENTION_MODELS = ["mistral", "mixtral", "qwen2", "qwen_moe", "starcoder2"]

        if len(self.all_generative_model_classes) == 0:
            self.skipTest(f"No generative model classes for {self.__class__.__name__}")

        for model_class in self.all_generative_model_classes:
            if model_class._supports_sdpa:
                self.skipTest(reason="Model architecture does not support attentions")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            if config.model_type not in WINDOW_ATTENTION_MODELS:
                self.skipTest(f"{config.model_type} does not use window attention")

            config.sliding_window = 2

            dummy_input = inputs_dict[model_class.main_input_name]
            attention_mask = inputs_dict["attention_mask"]

            self.assertTrue(dummy_input.ndim == 2)
            self.assertTrue(dummy_input.shape[1] > 6)

            with tempfile.TemporaryDirectory() as tmpdir:
                with torch.device(torch_device):
                    model_eager = AutoModelForCausalLM.from_config(
                        config, attn_implementation="eager", torch_dtype=torch.float32
                    )

                model_eager.save_pretrained(tmpdir)

                with torch.device(torch_device):
                    model_sdpa = AutoModelForCausalLM.from_pretrained(
                        tmpdir, attn_implementation="sdpa", torch_dtype=torch.float32
                    )

                model_eager = model_eager.eval()
                model_sdpa = model_sdpa.eval()

                with torch.no_grad():
                    with sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        res_eager = model_eager(**inputs_dict, return_dict=False)[0]
                        res_sdpa = model_sdpa(**inputs_dict, return_dict=False)[0]

                # Only non-padding tokens are expected to match.
                self.assertTrue(
                    torch.allclose(res_eager[attention_mask == 1], res_sdpa[attention_mask == 1], rtol=1e-4, atol=1e-4)
                )

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_flash_attn_2_can_dispatch_composite_models(self):
        """
        Tests if composite models can dispatch on FA2 if the sub-models support FA2.
        The tests is needed as we handle differently composite models and we cannot check them
        with above tests. If any of the sub-models does not support FA2, we'll raise an error when dispatching
        that particular sub-model. Otherwise we dispatch safely in all sub-models, where "sub-models" are specific
        backbone models (LM/vision/audio/etc)
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        torch_dtype = torch.float16
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            if not self._is_composite:
                self.skipTest("This model is not a composte model!")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)

                supports_fa2_all_modules = all(
                    module._supports_flash_attn_2
                    for name, module in model.named_modules()
                    if isinstance(module, PreTrainedModel) and name != ""
                )
                if not supports_fa2_all_modules:
                    with self.assertRaises(ValueError):
                        model_fa2 = model_class.from_pretrained(
                            tmpdirname, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
                        )
                else:
                    model_fa2 = model_class.from_pretrained(
                        tmpdirname, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
                    )
                    for key in model_fa2.config:
                        if isinstance(getattr(model_fa2.config, key), PretrainedConfig):
                            sub_config = getattr(model_fa2.config, key)
                            self.assertTrue(sub_config._attn_implementation == "flash_attention_2")

                    has_fa2 = False
                    for name, submodule in model_fa2.named_modules():
                        class_name = submodule.__class__.__name__
                        if "FlashAttention" in class_name:
                            has_fa2 = True
                            break
                    if not has_fa2:
                        raise ValueError("The FA2 model should have FA2 layers")

    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                batch_size = dummy_attention_mask.shape[0]

                is_padding_right = dummy_attention_mask[:, -1].sum().item() != batch_size

                # To avoid errors with padding_side=="right"
                if is_padding_right:
                    dummy_attention_mask = torch.ones_like(dummy_input)

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                    load_in_4bit=True,
                )

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                if model.config.is_encoder_decoder:
                    dummy_decoder_input_ids = inputs_dict["decoder_input_ids"]
                    dummy_decoder_attention_mask = inputs_dict["decoder_attention_mask"]

                    _ = model(dummy_input, decoder_input_ids=dummy_decoder_input_ids)
                    # with attention mask
                    _ = model(
                        dummy_input,
                        attention_mask=dummy_attention_mask,
                        decoder_input_ids=dummy_decoder_input_ids,
                        decoder_attention_mask=dummy_decoder_attention_mask,
                    )
                else:
                    _ = model(dummy_input)
                    # with attention mask
                    _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        max_new_tokens = 30

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if 0 not in inputs_dict.get("attention_mask", []) or "attention_mask" not in inputs_dict:
                self.skipTest("Model dummy inputs should contain padding in their attention mask")

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # ensure left padding, to adapt for some models
                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                inputs_dict["input_ids"][~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                        low_cpu_mem_usage=True,
                    )
                    .to(torch_device)
                    .eval()
                )

                # flatten
                padfree_inputs_dict = {
                    k: v[dummy_attention_mask.bool()].unsqueeze(0)
                    for k, v in inputs_dict.items()
                    if not k == "attention_mask"
                }
                # add position_ids
                padfree_inputs_dict["position_ids"] = (
                    torch.cat([torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()])
                    .long()
                    .unsqueeze(0)
                    .to(torch_device)
                )

                res_padded = model(**inputs_dict)
                res_padfree = model(**padfree_inputs_dict)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                torch.testing.assert_close(logits_padded.argmax(-1), logits_padfree.argmax(-1), atol=0, rtol=0)
                # acceptable numerical instability
                tol = torch.finfo(torch.float16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, atol=tol, rtol=tol)

    @is_pt_tf_cross_test
    def test_tf_from_pt_safetensors(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            tf_model_class_name = "TF" + model_class.__name__  # Add the "TF" at the beginning
            if not hasattr(transformers, tf_model_class_name):
                self.skipTest(reason="transformers does not have this model in TF version yet")

            tf_model_class = getattr(transformers, tf_model_class_name)

            pt_model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname, safe_serialization=True)
                tf_model_1 = tf_model_class.from_pretrained(tmpdirname, from_pt=True)

                pt_model.save_pretrained(tmpdirname, safe_serialization=False)
                tf_model_2 = tf_model_class.from_pretrained(tmpdirname, from_pt=True)

                # Check models are equal
                for p1, p2 in zip(tf_model_1.weights, tf_model_2.weights):
                    self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @is_pt_flax_cross_test
    def test_flax_from_pt_safetensors(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            flax_model_class_name = "Flax" + model_class.__name__  # Add the "Flax at the beginning
            if not hasattr(transformers, flax_model_class_name):
                self.skipTest(reason="transformers does not have this model in Flax version yet")

            flax_model_class = getattr(transformers, flax_model_class_name)

            pt_model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname, safe_serialization=True)
                flax_model_1 = flax_model_class.from_pretrained(tmpdirname, from_pt=True)

                pt_model.save_pretrained(tmpdirname, safe_serialization=False)
                flax_model_2 = flax_model_class.from_pretrained(tmpdirname, from_pt=True)

                # Check models are equal
                self.assertTrue(check_models_equal(flax_model_1, flax_model_2))

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_from_config(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            # TODO: to change it in the future with other relevant auto classes
            fa2_model = model_class._from_config(
                config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            ).to(torch_device)

            dummy_input = torch.LongTensor([[0, 2, 3, 4], [0, 2, 3, 4]]).to(torch_device)
            dummy_attention_mask = torch.LongTensor([[1, 1, 1, 1], [0, 1, 1, 1]]).to(torch_device)

            fa2_correctly_converted = False

            for _, module in fa2_model.named_modules():
                if "FlashAttention" in module.__class__.__name__:
                    fa2_correctly_converted = True
                    break

            self.assertTrue(fa2_correctly_converted)

            _ = fa2_model(input_ids=dummy_input, attention_mask=dummy_attention_mask)

            with tempfile.TemporaryDirectory() as tmpdirname:
                fa2_model.save_pretrained(tmpdirname)

                model_from_pretrained = model_class.from_pretrained(tmpdirname)

                self.assertTrue(model_from_pretrained.config._attn_implementation != "flash_attention_2")

                fa2_correctly_converted = False

                for _, module in model_from_pretrained.named_modules():
                    if "FlashAttention" in module.__class__.__name__:
                        fa2_correctly_converted = True
                        break

                self.assertFalse(fa2_correctly_converted)

    def _get_custom_4d_mask_test_data(self):
        # Sequence in which all but the last token is the same
        input_ids = torch.tensor(
            [[10, 11, 12, 13], [10, 11, 12, 14], [10, 11, 12, 15]], device=torch_device, dtype=torch.int64
        )
        position_ids = torch.tensor([[0, 1, 2, 3]] * 3, device=torch_device, dtype=torch.int64)

        # Combining common prefix with the unique ending tokens:
        input_ids_shared_prefix = torch.cat([input_ids[0][:-1], input_ids[:, -1]]).unsqueeze(0)

        # Creating a 4D mask where each of the last 3 tokens do not attend to each other.
        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1],
                    ]
                ]
            ],
        )
        # inverting the attention mask
        mask_dtype = torch.float32
        min_dtype = torch.finfo(mask_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=mask_dtype, device=torch_device) * min_dtype

        # Creating a position_ids tensor. note the repeating figures in the end.
        position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 3, 3]], device=torch_device, dtype=torch.int64)

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_custom_4d_attention_mask(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if len(self.all_generative_model_classes) == 0:
            self.skipTest(
                reason="Model architecture has no generative classes, and thus not necessarily supporting 4D masks"
            )

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(f"{model_class.__name__} is not guaranteed to work with custom 4D attention masks")
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            if getattr(config, "sliding_window", 0) is not None and getattr(config, "sliding_window", 0) > 0:
                self.skipTest(f"{model_class.__name__} with sliding window attention is not supported by this test")
            model = model_class(config).to(device=torch_device, dtype=torch.float32)
            set_model_for_less_flaky_test(model)

            (
                input_ids,
                position_ids,
                input_ids_shared_prefix,
                mask_shared_prefix,
                position_ids_shared_prefix,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(input_ids, position_ids=position_ids).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids_shared_prefix,
                attention_mask=mask_shared_prefix,
                position_ids=position_ids_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    @slow
    @require_torch_accelerator
    def test_torch_compile_for_training(self):
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        if not hasattr(self, "_torch_compile_train_cls"):
            self.skipTest(f"{self.__class__.__name__} doesn't have the attribute `_torch_compile_train_cls`.")

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cls = self._torch_compile_train_cls
        model = cls(config).to(torch_device)

        inputs = {
            "input_ids": torch.randint(low=1, high=model.config.vocab_size, size=(2, 10), device=torch_device),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                dtype=torch.int64,
                device=torch_device,
            ),
            "position_ids": torch.arange(0, 10, device=torch_device).unsqueeze(0),
            "labels": torch.randint(low=1, high=model.config.vocab_size, size=(2, 10), device=torch_device),
        }

        # eager backward
        set_seed(42)
        loss = model(**inputs).loss
        loss.backward()

        params = {name: param.grad.clone().detach().cpu() for name, param in model.named_parameters()}
        model.zero_grad()
        del loss

        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        # forward compilation
        set_seed(42)
        loss = model(**inputs).loss
        # backward compilation
        loss.backward()
        # check grad matches
        for name, param in model._orig_mod.named_parameters():
            torch.testing.assert_close(param.grad.detach().cpu(), params[name], rtol=1e-4, atol=1e-4)

    def test_forward_with_num_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            if "num_logits_to_keep" not in set(inspect.signature(model_class.forward).parameters.keys()):
                self.skipTest(reason="This model does not support `num_logits_to_keep` argument.")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            batch_size, sequence_length = inputs["input_ids"].shape
            vocab_size = config.get_text_config().vocab_size
            model = model_class(config).to(device=torch_device).eval()
            # some models have labels but `num_logits_to_keep` should not be used in train mode
            _ = inputs.pop("labels", None)

            # num_logits_to_keep=0 is a special case meaning "keep all logits"
            all_logits = model(**inputs, num_logits_to_keep=0).logits
            last_token_logits = model(**inputs, num_logits_to_keep=1).logits

            # Assert all shapes are correct
            self.assertEqual(tuple(all_logits.shape), (batch_size, sequence_length, vocab_size))
            self.assertEqual(tuple(last_token_logits.shape), (batch_size, 1, vocab_size))

            # Assert the last tokens are actually the same (except for the natural fluctuation due to order of FP ops)
            self.assertTrue(torch.allclose(all_logits[:, -1:, :], last_token_logits, atol=1e-5))

    @require_torch_gpu
    def test_flex_attention_with_grads(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flex_attn:
                self.skipTest(reason="This model does not support flex attention")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"
            model = model_class(config).to(device=torch_device, dtype=torch.float16)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(inputs_dict["input_ids"].to(torch_device))


global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
    attn_mask[:, 0] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
