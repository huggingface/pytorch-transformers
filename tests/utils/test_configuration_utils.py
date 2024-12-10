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
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
import warnings
from pathlib import Path

from huggingface_hub import HfFolder
from requests.exceptions import HTTPError

from transformers import AutoConfig, BertConfig, GPT2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import TOKEN, TemporaryHubRepo, is_staging_test


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402


config_common_kwargs = {
    "return_dict": False,
    "output_hidden_states": True,
    "output_attentions": True,
    "torchscript": True,
    "torch_dtype": "float16",
    "use_bfloat16": True,
    "tf_legacy_loss": True,
    "pruned_heads": {"a": 1},
    "tie_word_embeddings": False,
    "is_decoder": True,
    "cross_attention_hidden_size": 128,
    "add_cross_attention": True,
    "tie_encoder_decoder": True,
    "max_length": 50,
    "min_length": 3,
    "do_sample": True,
    "early_stopping": True,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 0.5,
    "temperature": 2.0,
    "top_k": 10,
    "top_p": 0.7,
    "typical_p": 0.2,
    "repetition_penalty": 0.8,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 5,
    "bad_words_ids": [1, 2, 3],
    "num_return_sequences": 3,
    "chunk_size_feed_forward": 5,
    "output_scores": True,
    "return_dict_in_generate": True,
    "forced_bos_token_id": 2,
    "forced_eos_token_id": 3,
    "remove_invalid_values": True,
    "architectures": ["BertModel"],
    "finetuning_task": "translation",
    "id2label": {0: "label"},
    "label2id": {"label": "0"},
    "tokenizer_class": "BertTokenizerFast",
    "prefix": "prefix",
    "bos_token_id": 6,
    "pad_token_id": 7,
    "eos_token_id": 8,
    "sep_token_id": 9,
    "decoder_start_token_id": 10,
    "exponential_decay_length_penalty": (5, 1.01),
    "suppress_tokens": [0, 1],
    "begin_suppress_tokens": 2,
    "task_specific_params": {"translation": "some_params"},
    "problem_type": "regression",
}


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_dynamic_config(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomConfig.register_for_auto_class()
            config = CustomConfig(attribute=42)

            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(config.auto_map, {"AutoConfig": "custom_configuration.CustomConfig"})

            new_config = AutoConfig.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            # Can't make an isinstance check because the new_config is from the FakeConfig class of a dynamic module
            self.assertEqual(new_config.__class__.__name__, "CustomConfig")
            self.assertEqual(new_config.attribute, 42)


class ConfigTestUtils(unittest.TestCase):
    def test_config_from_string(self):
        c = GPT2Config()

        # attempt to modify each of int/float/bool/str config records and verify they were updated
        n_embd = c.n_embd + 1  # int
        resid_pdrop = c.resid_pdrop + 1.0  # float
        scale_attn_weights = not c.scale_attn_weights  # bool
        summary_type = c.summary_type + "foo"  # str
        c.update_from_string(
            f"n_embd={n_embd},resid_pdrop={resid_pdrop},scale_attn_weights={scale_attn_weights},summary_type={summary_type}"
        )
        self.assertEqual(n_embd, c.n_embd, "mismatch for key: n_embd")
        self.assertEqual(resid_pdrop, c.resid_pdrop, "mismatch for key: resid_pdrop")
        self.assertEqual(scale_attn_weights, c.scale_attn_weights, "mismatch for key: scale_attn_weights")
        self.assertEqual(summary_type, c.summary_type, "mismatch for key: summary_type")

    def test_config_common_kwargs_is_complete(self):
        base_config = PretrainedConfig()
        missing_keys = [key for key in base_config.__dict__ if key not in config_common_kwargs]
        # If this part of the test fails, you have arguments to addin config_common_kwargs above.
        self.assertListEqual(
            missing_keys,
            [
                "is_encoder_decoder",
                "_name_or_path",
                "_commit_hash",
                "_attn_implementation_internal",
                "_attn_implementation_autoset",
                "transformers_version",
            ],
        )
        keys_with_defaults = [key for key, value in config_common_kwargs.items() if value == getattr(base_config, key)]
        if len(keys_with_defaults) > 0:
            raise ValueError(
                "The following keys are set with the default values in"
                " `test_configuration_common.config_common_kwargs` pick another value for them:"
                f" {', '.join(keys_with_defaults)}."
            )

    def test_nested_config_load_from_dict(self):
        config = AutoConfig.from_pretrained(
            "hf-internal-testing/tiny-random-CLIPModel", text_config={"num_hidden_layers": 2}
        )
        self.assertNotIsInstance(config.text_config, dict)
        self.assertEqual(config.text_config.__class__.__name__, "CLIPTextConfig")

    def test_from_pretrained_subfolder(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert-subfolder")
        self.assertIsNotNone(config)

        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert-subfolder", subfolder="bert")
        self.assertIsNotNone(config)

    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_local_versioning(self):
        configuration = AutoConfig.from_pretrained("google-bert/bert-base-cased")
        configuration.configuration_files = ["config.4.0.0.json"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            configuration.save_pretrained(tmp_dir)
            configuration.hidden_size = 2
            json.dump(configuration.to_dict(), open(os.path.join(tmp_dir, "config.4.0.0.json"), "w"))

            # This should pick the new configuration file as the version of Transformers is > 4.0.0
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 2)

            # Will need to be adjusted if we reach v42 and this test is still here.
            # Should pick the old configuration file as the version of Transformers is < 4.42.0
            configuration.configuration_files = ["config.42.0.0.json"]
            configuration.hidden_size = 768
            configuration.save_pretrained(tmp_dir)
            shutil.move(os.path.join(tmp_dir, "config.4.0.0.json"), os.path.join(tmp_dir, "config.42.0.0.json"))
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 768)

    def test_repo_versioning_before(self):
        # This repo has two configuration files, one for v4.0.0 and above with a different hidden size.
        repo = "hf-internal-testing/test-two-configs"

        import transformers as new_transformers

        new_transformers.configuration_utils.__version__ = "v4.0.0"
        new_configuration, kwargs = new_transformers.models.auto.AutoConfig.from_pretrained(
            repo, return_unused_kwargs=True
        )
        self.assertEqual(new_configuration.hidden_size, 2)
        # This checks `_configuration_file` ia not kept in the kwargs by mistake.
        self.assertDictEqual(kwargs, {})

        # Testing an older version by monkey-patching the version in the module it's used.
        import transformers as old_transformers

        old_transformers.configuration_utils.__version__ = "v3.0.0"
        old_configuration = old_transformers.models.auto.AutoConfig.from_pretrained(repo)
        self.assertEqual(old_configuration.hidden_size, 768)

    def test_saving_config_with_custom_generation_kwargs_raises_warning(self):
        config = BertConfig(min_length=3)  # `min_length = 3` is a non-default generation kwarg
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertWarns(UserWarning) as cm:
                config.save_pretrained(tmp_dir)
            self.assertIn("min_length", str(cm.warning))

    def test_get_non_default_generation_parameters(self):
        config = BertConfig()
        self.assertFalse(len(config._get_non_default_generation_parameters()) > 0)
        config = BertConfig(min_length=3)
        self.assertTrue(len(config._get_non_default_generation_parameters()) > 0)
        config = BertConfig(min_length=0)  # `min_length = 0` is a default generation kwarg
        self.assertFalse(len(config._get_non_default_generation_parameters()) > 0)

    def test_loading_config_do_not_raise_future_warnings(self):
        """Regression test for https://github.com/huggingface/transformers/issues/31002."""
        # Loading config should not raise a FutureWarning. It was the case before.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PretrainedConfig.from_pretrained("bert-base-uncased")
