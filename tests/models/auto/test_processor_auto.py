# coding=utf-8
# Copyright 2021 the HuggingFace Inc. team.
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
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from huggingface_hub import HfFolder, Repository

import transformers
from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    BertTokenizer,
    ProcessorMixin,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from transformers.testing_utils import TOKEN, TemporaryHubRepo, get_tests_dir, is_staging_test
from transformers.tokenization_utils import TOKENIZER_CONFIG_FILE
from transformers.utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, is_tokenizers_available


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402
from test_module.custom_processing import CustomProcessor  # noqa E402
from test_module.custom_tokenization import CustomTokenizer  # noqa E402


SAMPLE_PROCESSOR_CONFIG = get_tests_dir("fixtures/dummy_feature_extractor_config.json")
SAMPLE_VOCAB = get_tests_dir("fixtures/vocab.json")
SAMPLE_PROCESSOR_CONFIG_DIR = get_tests_dir("fixtures")


class AutoFeatureExtractorTest(unittest.TestCase):
    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "bla", "blou"]

    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_processor_from_model_shortcut(self):
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_repo(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            processor.save_pretrained(tmpdirname)

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_extractor_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # copy relevant files
            copyfile(SAMPLE_PROCESSOR_CONFIG, os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME))
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, "vocab.json"))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            if not os.path.isfile(os.path.join(tmpdirname, PROCESSOR_NAME)):
                # create one manually in order to perform this test's objective
                config_dict = {"processor_class": "Wav2Vec2Processor"}
                with open(os.path.join(tmpdirname, PROCESSOR_NAME), "w") as fp:
                    json.dump(config_dict, fp)

            # drop `processor_class` in tokenizer config
            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "r") as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")

            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_feat_extr_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            if os.path.isfile(os.path.join(tmpdirname, PROCESSOR_NAME)):
                # drop `processor_class` in processor
                with open(os.path.join(tmpdirname, PROCESSOR_NAME), "r") as f:
                    config_dict = json.load(f)
                    config_dict.pop("processor_class")

                with open(os.path.join(tmpdirname, PROCESSOR_NAME), "w") as f:
                    f.write(json.dumps(config_dict))

            # drop `processor_class` in tokenizer
            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "r") as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")

            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_tokenizer_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            if os.path.isfile(os.path.join(tmpdirname, PROCESSOR_NAME)):
                # drop `processor_class` in processor
                with open(os.path.join(tmpdirname, PROCESSOR_NAME), "r") as f:
                    config_dict = json.load(f)
                    config_dict.pop("processor_class")

                with open(os.path.join(tmpdirname, PROCESSOR_NAME), "w") as f:
                    f.write(json.dumps(config_dict))

            # drop `processor_class` in feature extractor
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "r") as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")

            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_model_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config(processor_class="Wav2Vec2Processor")
            model_config.save_pretrained(tmpdirname)
            # copy relevant files
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, "vocab.json"))
            # create emtpy sample processor
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "w") as f:
                f.write("{}")

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_from_pretrained_dynamic_processor(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            processor = AutoProcessor.from_pretrained("hf-internal-testing/test_dynamic_processor")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor", trust_remote_code=False
            )

        processor = AutoProcessor.from_pretrained("hf-internal-testing/test_dynamic_processor", trust_remote_code=True)
        self.assertTrue(processor.special_attribute_present)
        self.assertEqual(processor.__class__.__name__, "NewProcessor")

        feature_extractor = processor.feature_extractor
        self.assertTrue(feature_extractor.special_attribute_present)
        self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")

        tokenizer = processor.tokenizer
        self.assertTrue(tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")

            # Test we can also load the slow version
            new_processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor", trust_remote_code=True, use_fast=False
            )
            new_tokenizer = new_processor.tokenizer
            self.assertTrue(new_tokenizer.special_attribute_present)
            self.assertEqual(new_tokenizer.__class__.__name__, "NewTokenizer")
        else:
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")

    def test_new_processor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, CustomFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            AutoProcessor.register(CustomConfig, CustomProcessor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoProcessor.register(Wav2Vec2Config, Wav2Vec2Processor)

            # Now that the config is registered, it can be used as any other config with the auto-API
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)

            with tempfile.TemporaryDirectory() as tmp_dir:
                vocab_file = os.path.join(tmp_dir, "vocab.txt")
                with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                    vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
                tokenizer = CustomTokenizer(vocab_file)

            processor = CustomProcessor(feature_extractor, tokenizer)

            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(tmp_dir)
                new_processor = AutoProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_processor, CustomProcessor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_processor_conflict(self):
        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            special_attribute_present = False

        class NewTokenizer(BertTokenizer):
            special_attribute_present = False

        class NewProcessor(ProcessorMixin):
            feature_extractor_class = "AutoFeatureExtractor"
            tokenizer_class = "AutoTokenizer"
            special_attribute_present = False

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoProcessor.register(CustomConfig, NewProcessor)
            # If remote code is not set, the default is to use local classes.
            processor = AutoProcessor.from_pretrained("hf-internal-testing/test_dynamic_processor")
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertFalse(processor.special_attribute_present)
            self.assertFalse(processor.feature_extractor.special_attribute_present)
            self.assertFalse(processor.tokenizer.special_attribute_present)

            # If remote code is disabled, we load the local ones.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor", trust_remote_code=False
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertFalse(processor.special_attribute_present)
            self.assertFalse(processor.feature_extractor.special_attribute_present)
            self.assertFalse(processor.tokenizer.special_attribute_present)

            # If remote is enabled, we load from the Hub.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor", trust_remote_code=True
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertTrue(processor.special_attribute_present)
            self.assertTrue(processor.feature_extractor.special_attribute_present)
            self.assertTrue(processor.tokenizer.special_attribute_present)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_processor_with_extra_attributes(self):
        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            pass

        class NewTokenizer(BertTokenizer):
            pass

        class NewProcessor(ProcessorMixin):
            feature_extractor_class = "AutoFeatureExtractor"
            tokenizer_class = "AutoTokenizer"

            def __init__(self, feature_extractor, tokenizer, processor_attr_1=1, processor_attr_2=True):
                super().__init__(feature_extractor, tokenizer)

                self.processor_attr_1 = processor_attr_1
                self.processor_attr_2 = processor_attr_2

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoProcessor.register(CustomConfig, NewProcessor)
            # If remote code is not set, the default is to use local classes.
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor", processor_attr_2=False
            )
            self.assertEqual(processor.__class__.__name__, "NewProcessor")
            self.assertEqual(processor.processor_attr_1, 1)
            self.assertEqual(processor.processor_attr_2, False)
        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_auto_processor_creates_tokenizer(self):
        processor = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertEqual(processor.__class__.__name__, "BertTokenizerFast")

    def test_auto_processor_creates_image_processor(self):
        processor = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-convnext")
        self.assertEqual(processor.__class__.__name__, "ConvNextImageProcessor")


@is_staging_test
class ProcessorPushToHubTester(unittest.TestCase):
    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "bla", "blou"]

    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            processor = Wav2Vec2Processor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_processor = Wav2Vec2Processor.from_pretrained(tmp_repo.repo_id)
            for k, v in processor.feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_processor.feature_extractor, k))
            self.assertDictEqual(new_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            processor = Wav2Vec2Processor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(
                    tmp_dir,
                    repo_id=tmp_repo.repo_id,
                    push_to_hub=True,
                    token=self._token,
                )

            new_processor = Wav2Vec2Processor.from_pretrained(tmp_repo.repo_id)
            for k, v in processor.feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_processor.feature_extractor, k))
            self.assertDictEqual(new_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())

    def test_push_to_hub_dynamic_processor(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomFeatureExtractor.register_for_auto_class()
            CustomTokenizer.register_for_auto_class()
            CustomProcessor.register_for_auto_class()

            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)

            with tempfile.TemporaryDirectory() as tmp_dir:
                vocab_file = os.path.join(tmp_dir, "vocab.txt")
                with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                    vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
                tokenizer = CustomTokenizer(vocab_file)

            processor = CustomProcessor(feature_extractor, tokenizer)

            with tempfile.TemporaryDirectory() as tmp_dir:
                repo = Repository(tmp_dir, clone_from=tmp_repo, token=self._token)
                processor.save_pretrained(tmp_dir)

                # This has added the proper auto_map field to the feature extractor config
                self.assertDictEqual(
                    processor.feature_extractor.auto_map,
                    {
                        "AutoFeatureExtractor": "custom_feature_extraction.CustomFeatureExtractor",
                        "AutoProcessor": "custom_processing.CustomProcessor",
                    },
                )

                # This has added the proper auto_map field to the tokenizer config
                with open(os.path.join(tmp_dir, "tokenizer_config.json")) as f:
                    tokenizer_config = json.load(f)
                self.assertDictEqual(
                    tokenizer_config["auto_map"],
                    {
                        "AutoTokenizer": ["custom_tokenization.CustomTokenizer", None],
                        "AutoProcessor": "custom_processing.CustomProcessor",
                    },
                )

                # The code has been copied from fixtures
                self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_feature_extraction.py")))
                self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_tokenization.py")))
                self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "custom_processing.py")))

                repo.push_to_hub()

                new_processor = AutoProcessor.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
                # Can't make an isinstance check because the new_processor is from the CustomProcessor class of a dynamic module
                self.assertEqual(new_processor.__class__.__name__, "CustomProcessor")
