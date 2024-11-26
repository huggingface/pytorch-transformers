# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import pytest
from huggingface_hub import Repository

import transformers
from transformers import BertConfig, GPT2Model, is_safetensors_available, is_torch_available
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.testing_utils import (
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    RequestCounter,
    require_torch,
    slow,
)

from ..bert.test_modeling_bert import BertModelTester


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402


if is_torch_available():
    import torch
    from test_module.custom_modeling import CustomModel

    from transformers import (
        AutoBackbone,
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
        BertForMaskedLM,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertModel,
        FunnelBaseModel,
        FunnelModel,
        GenerationMixin,
        GPT2Config,
        GPT2LMHeadModel,
        ResNetBackbone,
        RobertaForMaskedLM,
        T5Config,
        T5ForConditionalGeneration,
        TapasConfig,
        TapasForQuestionAnswering,
        TimmBackbone,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
    )


@require_torch
class AutoModelTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModel.from_pretrained(model_name)
        model, loading_info = AutoModel.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertModel)

        self.assertEqual(len(loading_info["missing_keys"]), 0)
        # When using PyTorch checkpoint, the expected value is `8`. With `safetensors` checkpoint (if it is
        # installed), the expected value becomes `7`.
        EXPECTED_NUM_OF_UNEXPECTED_KEYS = 7 if is_safetensors_available() else 8
        self.assertEqual(len(loading_info["unexpected_keys"]), EXPECTED_NUM_OF_UNEXPECTED_KEYS)
        self.assertEqual(len(loading_info["mismatched_keys"]), 0)
        self.assertEqual(len(loading_info["error_msgs"]), 0)

    @slow
    def test_model_for_pretraining_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelForPreTraining.from_pretrained(model_name)
        model, loading_info = AutoModelForPreTraining.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForPreTraining)
        # Only one value should not be initialized and in the missing keys.
        for key, value in loading_info.items():
            self.assertEqual(len(value), 0)

    @slow
    def test_lmhead_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelWithLMHead.from_pretrained(model_name)
        model, loading_info = AutoModelWithLMHead.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_causal_lm(self):
        model_name = "openai-community/gpt2"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, GPT2Config)

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model, loading_info = AutoModelForCausalLM.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, GPT2LMHeadModel)

    @slow
    def test_model_for_masked_lm(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model, loading_info = AutoModelForMaskedLM.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_encoder_decoder_lm(self):
        model_name = "google-t5/t5-base"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, T5Config)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, T5ForConditionalGeneration)

    @slow
    def test_sequence_classification_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForSequenceClassification)

    @slow
    def test_question_answering_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model, loading_info = AutoModelForQuestionAnswering.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForQuestionAnswering)

    @slow
    def test_table_question_answering_model_from_pretrained(self):
        model_name = "google/tapas-base"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, TapasConfig)

        model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)
        model, loading_info = AutoModelForTableQuestionAnswering.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, TapasForQuestionAnswering)

    @slow
    def test_token_classification_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model, loading_info = AutoModelForTokenClassification.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForTokenClassification)

    @slow
    def test_auto_backbone_timm_model_from_pretrained(self):
        # Configs can't be loaded for timm models
        model = AutoBackbone.from_pretrained("resnet18", use_timm_backbone=True)

        with pytest.raises(ValueError):
            # We can't pass output_loading_info=True as we're loading from timm
            AutoBackbone.from_pretrained("resnet18", use_timm_backbone=True, output_loading_info=True)

        self.assertIsNotNone(model)
        self.assertIsInstance(model, TimmBackbone)

        # Check kwargs are correctly passed to the backbone
        model = AutoBackbone.from_pretrained("resnet18", use_timm_backbone=True, out_indices=(-2, -1))
        self.assertEqual(model.out_indices, [-2, -1])

        # Check out_features cannot be passed to Timm backbones
        with self.assertRaises(ValueError):
            _ = AutoBackbone.from_pretrained("resnet18", use_timm_backbone=True, out_features=["stage1"])

    @slow
    def test_auto_backbone_from_pretrained(self):
        model = AutoBackbone.from_pretrained("microsoft/resnet-18")
        model, loading_info = AutoBackbone.from_pretrained("microsoft/resnet-18", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, ResNetBackbone)

        # Check kwargs are correctly passed to the backbone
        model = AutoBackbone.from_pretrained("microsoft/resnet-18", out_indices=[-2, -1])
        self.assertEqual(model.out_indices, [-2, -1])
        self.assertEqual(model.out_features, ["stage3", "stage4"])

        model = AutoBackbone.from_pretrained("microsoft/resnet-18", out_features=["stage2", "stage4"])
        self.assertEqual(model.out_indices, [2, 4])
        self.assertEqual(model.out_features, ["stage2", "stage4"])

    def test_from_pretrained_identifier(self):
        model = AutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(model, BertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_identifier_from_model_type(self):
        model = AutoModelWithLMHead.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(model, RobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_pretrained_with_tuple_values(self):
        # For the auto model mapping, FunnelConfig has two models: FunnelModel and FunnelBaseModel
        model = AutoModel.from_pretrained("sgugger/funnel-random-tiny")
        self.assertIsInstance(model, FunnelModel)

        config = copy.deepcopy(model.config)
        config.architectures = ["FunnelBaseModel"]
        model = AutoModel.from_config(config)
        self.assertIsInstance(model, FunnelBaseModel)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(model, FunnelBaseModel)

    def test_from_pretrained_dynamic_model_local(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoModel.register(CustomConfig, CustomModel)

            config = CustomConfig(hidden_size=32)
            model = CustomModel(config)

            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)

                new_model = AutoModel.from_pretrained(tmp_dir, trust_remote_code=True)
                for p1, p2 in zip(model.parameters(), new_model.parameters()):
                    self.assertTrue(torch.equal(p1, p2))

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in MODEL_MAPPING._extra_content:
                del MODEL_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_model_distant(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=False)

        model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=True)
        self.assertEqual(model.__class__.__name__, "NewModel")

        # Test the dynamic module is loaded only once.
        reloaded_model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=True)
        self.assertIs(model.__class__, reloaded_model.__class__)

        # Test model can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded_model = AutoModel.from_pretrained(tmp_dir, trust_remote_code=True)

        self.assertEqual(reloaded_model.__class__.__name__, "NewModel")
        for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # The model file is cached in the snapshot directory. So the module file is not changed after dumping
        # to a temp dir. Because the revision of the module file is not changed.
        # Test the dynamic module is loaded only once if the module file is not changed.
        self.assertIs(model.__class__, reloaded_model.__class__)

        # Test the dynamic module is reloaded if we force it.
        reloaded_model = AutoModel.from_pretrained(
            "hf-internal-testing/test_dynamic_model", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(model.__class__, reloaded_model.__class__)

        # This one uses a relative import to a util file, this checks it is downloaded and used properly.
        model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model_with_util", trust_remote_code=True)
        self.assertEqual(model.__class__.__name__, "NewModel")

        # Test the dynamic module is loaded only once.
        reloaded_model = AutoModel.from_pretrained(
            "hf-internal-testing/test_dynamic_model_with_util", trust_remote_code=True
        )
        self.assertIs(model.__class__, reloaded_model.__class__)

        # Test model can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded_model = AutoModel.from_pretrained(tmp_dir, trust_remote_code=True)

        self.assertEqual(reloaded_model.__class__.__name__, "NewModel")
        for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # The model file is cached in the snapshot directory. So the module file is not changed after dumping
        # to a temp dir. Because the revision of the module file is not changed.
        # Test the dynamic module is loaded only once if the module file is not changed.
        self.assertIs(model.__class__, reloaded_model.__class__)

        # Test the dynamic module is reloaded if we force it.
        reloaded_model = AutoModel.from_pretrained(
            "hf-internal-testing/test_dynamic_model_with_util", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(model.__class__, reloaded_model.__class__)

    def test_from_pretrained_dynamic_model_distant_with_ref(self):
        model = AutoModel.from_pretrained("hf-internal-testing/ref_to_test_dynamic_model", trust_remote_code=True)
        self.assertEqual(model.__class__.__name__, "NewModel")

        # Test model can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded_model = AutoModel.from_pretrained(tmp_dir, trust_remote_code=True)

        self.assertEqual(reloaded_model.__class__.__name__, "NewModel")
        for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # This one uses a relative import to a util file, this checks it is downloaded and used properly.
        model = AutoModel.from_pretrained(
            "hf-internal-testing/ref_to_test_dynamic_model_with_util", trust_remote_code=True
        )
        self.assertEqual(model.__class__.__name__, "NewModel")

        # Test model can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded_model = AutoModel.from_pretrained(tmp_dir, trust_remote_code=True)

        self.assertEqual(reloaded_model.__class__.__name__, "NewModel")
        for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_from_pretrained_dynamic_model_with_period(self):
        # We used to have issues where repos with "." in the name would cause issues because the Python
        # import machinery would treat that as a directory separator, so we test that case

        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model_v1.0")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model_v1.0", trust_remote_code=False)

        model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model_v1.0", trust_remote_code=True)
        self.assertEqual(model.__class__.__name__, "NewModel")

        # Test that it works with a custom cache dir too
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModel.from_pretrained(
                "hf-internal-testing/test_dynamic_model_v1.0", trust_remote_code=True, cache_dir=tmp_dir
            )
            self.assertEqual(model.__class__.__name__, "NewModel")

    def test_new_model_registration(self):
        AutoConfig.register("custom", CustomConfig)

        auto_classes = [
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        ]

        try:
            for auto_class in auto_classes:
                with self.subTest(auto_class.__name__):
                    # Wrong config class will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, CustomModel)
                    auto_class.register(CustomConfig, CustomModel)
                    # Trying to register something existing in the Transformers library will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, BertModel)

                    # Now that the config is registered, it can be used as any other config with the auto-API
                    tiny_config = BertModelTester(self).get_config()
                    config = CustomConfig(**tiny_config.to_dict())
                    model = auto_class.from_config(config)
                    self.assertIsInstance(model, CustomModel)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model.save_pretrained(tmp_dir)
                        new_model = auto_class.from_pretrained(tmp_dir)
                        # The model is a CustomModel but from the new dynamically imported class.
                        self.assertIsInstance(new_model, CustomModel)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            for mapping in (
                MODEL_MAPPING,
                MODEL_FOR_PRETRAINING_MAPPING,
                MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_MASKED_LM_MAPPING,
            ):
                if CustomConfig in mapping._extra_content:
                    del mapping._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_model_conflict(self):
        class NewModelConfigLocal(BertConfig):
            model_type = "new-model"

        class NewModel(BertModel):
            config_class = NewModelConfigLocal

        try:
            AutoConfig.register("new-model", NewModelConfigLocal)
            AutoModel.register(NewModelConfigLocal, NewModel)
            # If remote code is not set, the default is to use local
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model")
            self.assertEqual(model.config.__class__.__name__, "NewModelConfigLocal")

            # If remote code is disabled, we load the local one.
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=False)
            self.assertEqual(model.config.__class__.__name__, "NewModelConfigLocal")

            # If remote is enabled, we load from the Hub
            model = AutoModel.from_pretrained("hf-internal-testing/test_dynamic_model", trust_remote_code=True)
            self.assertEqual(model.config.__class__.__name__, "NewModelConfig")

        finally:
            if "new-model" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["new-model"]
            if NewModelConfigLocal in MODEL_MAPPING._extra_content:
                del MODEL_MAPPING._extra_content[NewModelConfigLocal]

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoModel.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoModel.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_model_file_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named pytorch_model.bin",
        ):
            _ = AutoModel.from_pretrained("hf-internal-testing/config-no-model")

    def test_model_from_tf_suggestion(self):
        with self.assertRaisesRegex(EnvironmentError, "Use `from_tf=True` to load this model"):
            _ = AutoModel.from_pretrained("hf-internal-testing/tiny-bert-tf-only")

    def test_model_from_flax_suggestion(self):
        with self.assertRaisesRegex(EnvironmentError, "Use `from_flax=True` to load this model"):
            _ = AutoModel.from_pretrained("hf-internal-testing/tiny-bert-flax-only")

    def test_cached_model_has_minimum_calls_to_head(self):
        # Make sure we have cached the model.
        _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertEqual(counter["GET"], 0)
        self.assertEqual(counter["HEAD"], 1)
        self.assertEqual(counter.total_calls, 1)

        # With a sharded checkpoint
        _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
        with RequestCounter() as counter:
            _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
        self.assertEqual(counter["GET"], 0)
        self.assertEqual(counter["HEAD"], 1)
        self.assertEqual(counter.total_calls, 1)

    def test_attr_not_existing(self):
        from transformers.models.auto.auto_factory import _LazyAutoMapping

        _CONFIG_MAPPING_NAMES = OrderedDict([("bert", "BertConfig")])
        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "GhostModel")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)

        with pytest.raises(ValueError, match=r"Could not find GhostModel neither in .* nor in .*!"):
            _MODEL_MAPPING[BertConfig]

        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "BertModel")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)
        self.assertEqual(_MODEL_MAPPING[BertConfig], BertModel)

        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "GPT2Model")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)
        self.assertEqual(_MODEL_MAPPING[BertConfig], GPT2Model)

    def test_dynamic_saving_from_local_repo(self):
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as tmp_dir_out:
            _ = Repository(local_dir=tmp_dir, clone_from="hf-internal-testing/tiny-random-custom-architecture")
            model = AutoModelForCausalLM.from_pretrained(tmp_dir, trust_remote_code=True)
            model.save_pretrained(tmp_dir_out)
            _ = AutoModelForCausalLM.from_pretrained(tmp_dir_out, trust_remote_code=True)
            self.assertTrue((Path(tmp_dir_out) / "modeling_fake_custom.py").is_file())
            self.assertTrue((Path(tmp_dir_out) / "configuration_fake_custom.py").is_file())

    def test_custom_model_patched_generation_inheritance(self):
        """
        Tests that our inheritance patching for generate-compatible models works as expected. Without this feature,
        old Hub models lose the ability to call `generate`.
        """
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/test_dynamic_model_generation", trust_remote_code=True
        )
        self.assertTrue(model.__class__.__name__ == "NewModelForCausalLM")

        # It inherits from GenerationMixin. This means it can `generate`. Because `PreTrainedModel` is scheduled to
        # stop inheriting from `GenerationMixin` in v4.50, this check will fail if patching is not present.
        self.assertTrue(isinstance(model, GenerationMixin))
        # More precisely, it directly inherits from GenerationMixin. This check would fail prior to v4.45 (inheritance
        # patching was added in v4.45)
        self.assertTrue("GenerationMixin" in str(model.__class__.__bases__))
