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
import sys
import tempfile
import unittest
from pathlib import Path

import transformers
from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    CLIPConfig,
    CLIPImageProcessor,
    ViTImageProcessor,
    ViTImageProcessorFast,
)
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, require_torchvision, require_vision


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_image_processing import CustomImageProcessor  # noqa E402


class AutoImageProcessorTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_image_processor_from_model_shortcut(self):
        config = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_key(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / "preprocessor_config.json"
            config_tmpfile = Path(tmpdirname) / "config.json"
            json.dump(
                {"image_processor_type": "CLIPImageProcessor", "processor_class": "CLIPProcessor"},
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "clip"}, open(config_tmpfile, "w"))

            config = AutoImageProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_feature_extractor_key(self):
        # Ensure we can load the image processor from the feature extractor config
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / "preprocessor_config.json"
            config_tmpfile = Path(tmpdirname) / "config.json"
            json.dump(
                {"feature_extractor_type": "CLIPFeatureExtractor", "processor_class": "CLIPProcessor"},
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "clip"}, open(config_tmpfile, "w"))

            config = AutoImageProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = CLIPConfig()

            # Create a dummy config file with image_proceesor_type
            processor_tmpfile = Path(tmpdirname) / "preprocessor_config.json"
            config_tmpfile = Path(tmpdirname) / "config.json"
            json.dump(
                {"image_processor_type": "CLIPImageProcessor", "processor_class": "CLIPProcessor"},
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "clip"}, open(config_tmpfile, "w"))

            # remove image_processor_type to make sure config.json alone is enough to load image processor locally
            config_dict = AutoImageProcessor.from_pretrained(tmpdirname).to_dict()

            config_dict.pop("image_processor_type")
            config = CLIPImageProcessor(**config_dict)

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)

            config = AutoImageProcessor.from_pretrained(tmpdirname)

            # make sure private variable is not incorrectly saved
            dict_as_saved = json.loads(config.to_json_string())
            self.assertTrue("_processor_class" not in dict_as_saved)

        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / "preprocessor_config.json"
            json.dump(
                {"image_processor_type": "CLIPImageProcessor", "processor_class": "CLIPProcessor"},
                open(processor_tmpfile, "w"),
            )

            config = AutoImageProcessor.from_pretrained(processor_tmpfile)
            self.assertIsInstance(config, CLIPImageProcessor)

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "clip-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoImageProcessor.from_pretrained("clip-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoImageProcessor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_image_processor_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named preprocessor_config.json.",
        ):
            _ = AutoImageProcessor.from_pretrained("hf-internal-testing/config-no-model")

    @require_vision
    @require_torchvision
    def test_use_fast_selection(self):
        checkpoint = "hf-internal-testing/tiny-random-vit"

        # TODO: @yoni, change in v4.48 (when use_fast set to True by default)
        # Slow image processor is selected by default
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.assertIsInstance(image_processor, ViTImageProcessor)

        # Fast image processor is selected when use_fast=True
        image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        self.assertIsInstance(image_processor, ViTImageProcessorFast)

        # Slow image processor is selected when use_fast=False
        image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)
        self.assertIsInstance(image_processor, ViTImageProcessor)

    def test_from_pretrained_dynamic_image_processor(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            image_processor = AutoImageProcessor.from_pretrained("hf-internal-testing/test_dynamic_image_processor")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            image_processor = AutoImageProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=False
            )

        image_processor = AutoImageProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=True
        )
        self.assertEqual(image_processor.__class__.__name__, "NewImageProcessor")

        # Test the dynamic module is loaded only once.
        reloaded_image_processor = AutoImageProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=True
        )
        self.assertIs(image_processor.__class__, reloaded_image_processor.__class__)

        # Test image processor can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(tmp_dir)
            reloaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertEqual(reloaded_image_processor.__class__.__name__, "NewImageProcessor")

        # The image processor file is cached in the snapshot directory. So the module file is not changed after dumping
        # to a temp dir. Because the revision of the module file is not changed.
        # Test the dynamic module is loaded only once if the module file is not changed.
        self.assertIs(image_processor.__class__, reloaded_image_processor.__class__)

        # Test the dynamic module is reloaded if we force it.
        reloaded_image_processor = AutoImageProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(image_processor.__class__, reloaded_image_processor.__class__)

    def test_new_image_processor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoImageProcessor.register(CustomConfig, CustomImageProcessor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoImageProcessor.register(CLIPConfig, CLIPImageProcessor)

            with tempfile.TemporaryDirectory() as tmpdirname:
                processor_tmpfile = Path(tmpdirname) / "preprocessor_config.json"
                config_tmpfile = Path(tmpdirname) / "config.json"
                json.dump(
                    {"feature_extractor_type": "CLIPFeatureExtractor", "processor_class": "CLIPProcessor"},
                    open(processor_tmpfile, "w"),
                )
                json.dump({"model_type": "clip"}, open(config_tmpfile, "w"))

                image_processor = CustomImageProcessor.from_pretrained(tmpdirname)

            # Now that the config is registered, it can be used as any other config with the auto-API
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_processor.save_pretrained(tmp_dir)
                new_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_image_processor, CustomImageProcessor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in IMAGE_PROCESSOR_MAPPING._extra_content:
                del IMAGE_PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_image_processor_conflict(self):
        class NewImageProcessor(CLIPImageProcessor):
            is_local = True

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoImageProcessor.register(CustomConfig, NewImageProcessor)
            # If remote code is not set, the default is to use local
            image_processor = AutoImageProcessor.from_pretrained("hf-internal-testing/test_dynamic_image_processor")
            self.assertEqual(image_processor.__class__.__name__, "NewImageProcessor")
            self.assertTrue(image_processor.is_local)

            # If remote code is disabled, we load the local one.
            image_processor = AutoImageProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=False
            )
            self.assertEqual(image_processor.__class__.__name__, "NewImageProcessor")
            self.assertTrue(image_processor.is_local)

            # If remote is enabled, we load from the Hub
            image_processor = AutoImageProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=True
            )
            self.assertEqual(image_processor.__class__.__name__, "NewImageProcessor")
            self.assertTrue(not hasattr(image_processor, "is_local"))

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in IMAGE_PROCESSOR_MAPPING._extra_content:
                del IMAGE_PROCESSOR_MAPPING._extra_content[CustomConfig]
