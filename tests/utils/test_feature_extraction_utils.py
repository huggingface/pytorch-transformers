# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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

from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers.testing_utils import TOKEN, TemporaryHubRepo, get_tests_dir, is_staging_test


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402


SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir("fixtures")


class FeatureExtractorUtilTester(unittest.TestCase):
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
            # This check we did call the fake head request
            mock_head.assert_called()


@is_staging_test
class FeatureExtractorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            feature_extractor.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                feature_extractor.save_pretrained(
                    tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token
                )

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            feature_extractor.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                feature_extractor.save_pretrained(
                    tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token
                )

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_dynamic_feature_extractor(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomFeatureExtractor.register_for_auto_class()
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)

            feature_extractor.push_to_hub(tmp_repo.repo_id, token=self._token)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(
                feature_extractor.auto_map,
                {"AutoFeatureExtractor": "custom_feature_extraction.CustomFeatureExtractor"},
            )

            new_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            # Can't make an isinstance check because the new_feature_extractor is from the CustomFeatureExtractor class of a dynamic module
            self.assertEqual(new_feature_extractor.__class__.__name__, "CustomFeatureExtractor")
