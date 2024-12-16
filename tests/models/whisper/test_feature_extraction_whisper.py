# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import itertools
import os
import random
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from transformers import WhisperFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_torch_gpu
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


class WhisperFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=10,
        hop_length=160,
        chunk_length=8,
        padding_value=0.0,
        sampling_rate=4_000,
        return_attention_mask=False,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.feature_size = feature_size
        self.chunk_length = chunk_length
        self.hop_length = hop_length

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "hop_length": self.hop_length,
            "chunk_length": self.chunk_length,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                floats_list((x, self.feature_size))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


class WhisperFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = WhisperFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = WhisperFeatureExtractionTester(self)

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = feat_extract_first.mel_filters
        mel_2 = feat_extract_second.mel_filters
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = feat_extract_first.mel_filters
        mel_2 = feat_extract_second.mel_filters
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_from_pretrained_kwargs(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(
                tmpdirname, feature_size=2 * self.feat_extract_dict["feature_size"]
            )

        mel_1 = feat_extract_first.mel_filters
        mel_2 = feat_extract_second.mel_filters
        self.assertTrue(2 * mel_1.shape[1] == mel_2.shape[1])

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding="max_length", return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.nb_max_frames)
        self.assertTrue(input_features.shape[-2] == feature_extractor.feature_size)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test truncation required
        speech_inputs = [floats_list((1, x))[0] for x in range(200, (feature_extractor.n_samples + 500), 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        speech_inputs_truncated = [x[: feature_extractor.n_samples] for x in speech_inputs]
        np_speech_inputs_truncated = [np.asarray(speech_input) for speech_input in speech_inputs_truncated]

        encoded_sequences_1 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs_truncated, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    @require_torch
    def test_double_precision_pad(self):
        import torch

        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @require_torch_gpu
    @require_torch
    def test_torch_integration(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                0.1193, -0.0946, -0.1098, -0.0196, 0.0225, -0.0690, -0.1736, 0.0951,
                0.0971, -0.0817, -0.0702, 0.0162, 0.0260, 0.0017, -0.0192, -0.1678,
                0.0709, -0.1867, -0.0655, -0.0274, -0.0234, -0.1884, -0.0516, -0.0554,
                -0.0274, -0.1425, -0.1423, 0.0837, 0.0377, -0.0854
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").input_features

        self.assertEqual(input_features.shape, (1, 80, 3000))
        self.assertTrue(torch.allclose(input_features[0, 0, :30], EXPECTED_INPUT_FEATURES, atol=1e-4))

    @unittest.mock.patch("transformers.models.whisper.feature_extraction_whisper.is_torch_available", lambda: False)
    def test_numpy_integration(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = np.array(
            [
                0.1193, -0.0946, -0.1098, -0.0196, 0.0225, -0.0690, -0.1736, 0.0951,
                0.0971, -0.0817, -0.0702, 0.0162, 0.0260, 0.0017, -0.0192, -0.1678,
                0.0709, -0.1867, -0.0655, -0.0274, -0.0234, -0.1884, -0.0516, -0.0554,
                -0.0274, -0.1425, -0.1423, 0.0837, 0.0377, -0.0854
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="np").input_features
        self.assertEqual(input_features.shape, (1, 80, 3000))
        self.assertTrue(np.allclose(input_features[0, 0, :30], EXPECTED_INPUT_FEATURES, atol=1e-4))

    def test_zero_mean_unit_variance_normalization_trunc_np_longest(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = self._load_datasamples(1)[0]
        audio = ((audio - audio.min()) / (audio.max() - audio.min())) * 65535  # Rescale to [0, 65535] to show issue
        audio = feat_extract.zero_mean_unit_var_norm([audio], attention_mask=None)[0]

        self.assertTrue(np.all(np.mean(audio) < 1e-3))
        self.assertTrue(np.all(np.abs(np.var(audio) - 1) < 1e-3))

    @require_torch_gpu
    @require_torch
    def test_torch_integration_batch(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    0.1193, -0.0946, -0.1098, -0.0196, 0.0225, -0.0690, -0.1736, 0.0951,
                    0.0971, -0.0817, -0.0702, 0.0162, 0.0260, 0.0017, -0.0192, -0.1678,
                    0.0709, -0.1867, -0.0655, -0.0274, -0.0234, -0.1884, -0.0516, -0.0554,
                    -0.0274, -0.1425, -0.1423, 0.0837, 0.0377, -0.0854
                ],
                [
                    -0.4696, -0.0751, 0.0276, -0.0312, -0.0540, -0.0383, 0.1295, 0.0568,
                    -0.2071, -0.0548, 0.0389, -0.0316, -0.2346, -0.1068, -0.0322, 0.0475,
                    -0.1709, -0.0041, 0.0872, 0.0537, 0.0075, -0.0392, 0.0371, 0.0189,
                    -0.1522, -0.0270, 0.0744, 0.0738, -0.0245, -0.0667
                ],
                [
                    -0.2337, -0.0060, -0.0063, -0.2353, -0.0431, 0.1102, -0.1492, -0.0292,
                     0.0787, -0.0608, 0.0143, 0.0582, 0.0072, 0.0101, -0.0444, -0.1701,
                     -0.0064, -0.0027, -0.0826, -0.0730, -0.0099, -0.0762, -0.0170, 0.0446,
                     -0.1153, 0.0960, -0.0361, 0.0652, 0.1207, 0.0277
                ]
            ]
        )
        # fmt: on

        input_speech = self._load_datasamples(3)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").input_features
        self.assertEqual(input_features.shape, (3, 80, 3000))
        self.assertTrue(torch.allclose(input_features[:, 0, :30], EXPECTED_INPUT_FEATURES, atol=1e-4))
