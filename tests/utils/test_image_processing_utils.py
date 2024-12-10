# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder
from requests.exceptions import HTTPError

from transformers import AutoImageProcessor, ViTImageProcessor
from transformers.image_processing_utils import get_size_dict
from transformers.testing_utils import TOKEN, TemporaryHubRepo, get_tests_dir, is_staging_test


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_image_processing import CustomImageProcessor  # noqa E402


SAMPLE_IMAGE_PROCESSING_CONFIG_DIR = get_tests_dir("fixtures")


class ImageProcessorUtilTester(unittest.TestCase):
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = ViTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = ViTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_image_processor_from_pretrained_subfolder(self):
        with self.assertRaises(OSError):
            # config is in subfolder, the following should not work without specifying the subfolder
            _ = AutoImageProcessor.from_pretrained("hf-internal-testing/stable-diffusion-all-variants")

        config = AutoImageProcessor.from_pretrained(
            "hf-internal-testing/stable-diffusion-all-variants", subfolder="feature_extractor"
        )

        self.assertIsNotNone(config)


@is_staging_test
class ImageProcessorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            image_processor = ViTImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
            image_processor.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_image_processor = ViTImageProcessor.from_pretrained(tmp_repo.repo_id)
            for k, v in image_processor.__dict__.items():
                self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            image_processor = ViTImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_processor.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_image_processor = ViTImageProcessor.from_pretrained(tmp_repo.repo_id)
            for k, v in image_processor.__dict__.items():
                self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            image_processor = ViTImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
            image_processor.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_image_processor = ViTImageProcessor.from_pretrained(tmp_repo.repo_id)
            for k, v in image_processor.__dict__.items():
                self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            image_processor = ViTImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_processor.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_image_processor = ViTImageProcessor.from_pretrained(tmp_repo.repo_id)
            for k, v in image_processor.__dict__.items():
                self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_dynamic_image_processor(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomImageProcessor.register_for_auto_class()
            image_processor = CustomImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)

            image_processor.push_to_hub(tmp_repo.repo_id, token=self._token)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(
                image_processor.auto_map,
                {"AutoImageProcessor": "custom_image_processing.CustomImageProcessor"},
            )

            new_image_processor = AutoImageProcessor.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            # Can't make an isinstance check because the new_image_processor is from the CustomImageProcessor class of a dynamic module
            self.assertEqual(new_image_processor.__class__.__name__, "CustomImageProcessor")


class ImageProcessingUtilsTester(unittest.TestCase):
    def test_get_size_dict(self):
        # Test a dict with the wrong keys raises an error
        inputs = {"wrong_key": 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)

        inputs = {"height": 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)

        inputs = {"width": 224, "shortest_edge": 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)

        # Test a dict with the correct keys is returned as is
        inputs = {"height": 224, "width": 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, inputs)

        inputs = {"shortest_edge": 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, {"shortest_edge": 224})

        inputs = {"longest_edge": 224, "shortest_edge": 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, {"longest_edge": 224, "shortest_edge": 224})

        # Test a single int value which  represents (size, size)
        outputs = get_size_dict(224)
        self.assertEqual(outputs, {"height": 224, "width": 224})

        # Test a single int value which represents the shortest edge
        outputs = get_size_dict(224, default_to_square=False)
        self.assertEqual(outputs, {"shortest_edge": 224})

        # Test a tuple of ints which represents (height, width)
        outputs = get_size_dict((150, 200))
        self.assertEqual(outputs, {"height": 150, "width": 200})

        # Test a tuple of ints which represents (width, height)
        outputs = get_size_dict((150, 200), height_width_order=False)
        self.assertEqual(outputs, {"height": 200, "width": 150})

        # Test an int representing the shortest edge and max_size which represents the longest edge
        outputs = get_size_dict(224, max_size=256, default_to_square=False)
        self.assertEqual(outputs, {"shortest_edge": 224, "longest_edge": 256})

        # Test int with default_to_square=True and max_size fails
        with self.assertRaises(ValueError):
            get_size_dict(224, max_size=256, default_to_square=True)
