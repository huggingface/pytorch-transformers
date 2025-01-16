# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

from transformers import SeamlessM4TFeatureExtractor, is_speech_available
from transformers.testing_utils import check_json_file_has_correct_format, require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torch_available():
    import torch

global_rng = random.Random()


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
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


@require_torch
class SeamlessM4TFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=10,
        padding_value=0.0,
        sampling_rate=4_000,
        return_attention_mask=True,
        do_normalize=True,
        stride=2,
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
        self.stride = stride
        self.num_mel_bins = feature_size

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "num_mel_bins": self.num_mel_bins,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "stride": self.stride,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
        }

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTester.prepare_inputs_for_common
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


@require_torch
class SeamlessM4TFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = SeamlessM4TFeatureExtractor if is_speech_available() else None

    def setUp(self):
        self.feat_extract_tester = SeamlessM4TFeatureExtractionTester(self)

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertDictEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_call_numpy(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding=True, return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[0] == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size * feature_extractor.stride)

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

    def test_call_with_padded_input_not_multiple_of_stride(self):
        # same as test_call_numpy but with stride=6 and pad_to_multiple_of=8
        # the input sizes 800, 1400 and 200 are a multiple of pad_to_multiple_of but not a multiple of stride
        # therefore remainder = num_frames % self.stride will not be zero and must be subtracted from num_frames
        stride = 6
        pad_to_multiple_of = 8

        feature_extractor_args = self.feat_extract_tester.prepare_feat_extract_dict()
        feature_extractor_args["stride"] = stride
        feature_extractor = self.feature_extraction_class(**feature_extractor_args)

        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size and attention mask size
        output = feature_extractor(np_speech_inputs, pad_to_multiple_of=pad_to_multiple_of, return_tensors="np")
        input_features = output.input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[0] == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size * feature_extractor.stride)
        # same as test_attention_mask
        attention_mask = output.attention_mask
        self.assertTrue(attention_mask.ndim == 2)
        self.assertTrue(attention_mask.shape[0] == 3)
        self.assertTrue(attention_mask.shape[-1] == input_features.shape[1])

        # Test not batched input
        encoded_sequences_1 = feature_extractor(
            speech_inputs[0], pad_to_multiple_of=pad_to_multiple_of, return_tensors="np"
        ).input_features
        encoded_sequences_2 = feature_extractor(
            np_speech_inputs[0], pad_to_multiple_of=pad_to_multiple_of, return_tensors="np"
        ).input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feature_extractor(
            speech_inputs, pad_to_multiple_of=pad_to_multiple_of, return_tensors="np"
        ).input_features
        encoded_sequences_2 = feature_extractor(
            np_speech_inputs, pad_to_multiple_of=pad_to_multiple_of, return_tensors="np"
        ).input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feature_extractor(
            speech_inputs, pad_to_multiple_of=pad_to_multiple_of, return_tensors="np"
        ).input_features
        encoded_sequences_2 = feature_extractor(
            np_speech_inputs, pad_to_multiple_of=pad_to_multiple_of, return_tensors="np"
        ).input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_call_without_attention_mask(self):
        feature_extractor_args = self.feat_extract_tester.prepare_feat_extract_dict()
        feature_extractor = self.feature_extraction_class(**feature_extractor_args)

        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test attention mask when passing no attention mask to forward call
        output = feature_extractor(np_speech_inputs, padding=True, return_tensors="np", return_attention_mask=False)
        self.assertTrue("attention_mask" not in output)

        # Test attention mask when no attention mask by default
        feature_extractor_args["return_attention_mask"] = False
        feature_extractor = self.feature_extraction_class(**feature_extractor_args)
        output = feature_extractor(np_speech_inputs, padding=True, return_tensors="np", return_attention_mask=False)
        self.assertTrue("attention_mask" not in output)

    def test_attention_mask(self):
        # test attention mask has the right output shape
        feature_extractor_args = self.feat_extract_tester.prepare_feat_extract_dict()

        feature_extractor = self.feature_extraction_class(**feature_extractor_args)
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test attention mask when passing it to forward call
        output = feature_extractor(np_speech_inputs, padding=True, return_tensors="np")
        input_features = output.input_features

        attention_mask = output.attention_mask
        self.assertTrue(attention_mask.ndim == 2)
        self.assertTrue(attention_mask.shape[0] == 3)
        self.assertTrue(attention_mask.shape[-1] == input_features.shape[1])

    @require_torch
    def test_call_torch(self):
        import torch

        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        pt_speech_inputs = [torch.tensor(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(pt_speech_inputs, padding=True, return_tensors="pt").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[0] == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size * feature_extractor.stride)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors="pt").input_features
        encoded_sequences_2 = feature_extractor(pt_speech_inputs[0], return_tensors="pt").input_features
        torch.testing.assert_close(encoded_sequences_1, encoded_sequences_2, rtol=1e-3, atol=1e-3)

        # Test batched
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="pt").input_features
        encoded_sequences_2 = feature_extractor(pt_speech_inputs, return_tensors="pt").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            torch.testing.assert_close(enc_seq_1, enc_seq_2, rtol=1e-3, atol=1e-3)

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        pt_speech_inputs = torch.tensor(speech_inputs)
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="pt").input_features
        encoded_sequences_2 = feature_extractor(pt_speech_inputs, return_tensors="pt").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            torch.testing.assert_close(enc_seq_1, enc_seq_2, rtol=1e-3, atol=1e-3)

    @require_torch
    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_double_precision_pad
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

    def _load_datasample(self, id):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_sample = ds.sort("id")[id]["audio"]["array"]

        return torch.from_numpy(speech_sample).unsqueeze(0)

    def test_integration(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
            -1.5621, -1.4236, -1.3335, -1.3991, -1.2881, -1.1133, -0.9710, -0.8895,
            -0.8280, -0.7376, -0.7194, -0.6896, -0.6849, -0.6788, -0.6545, -0.6610,
            -0.6566, -0.5738, -0.5252, -0.5533, -0.5887, -0.6116, -0.5971, -0.4956,
            -0.2881, -0.1512,  0.0299,  0.1762,  0.2728,  0.2236
            ]
        )
        # fmt: on

        input_speech = self._load_datasample(10)
        feature_extractor = SeamlessM4TFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").input_features

        feature_extractor(input_speech, return_tensors="pt").input_features[0, 5, :30]
        self.assertEqual(input_features.shape, (1, 279, 160))
        torch.testing.assert_close(input_features[0, 5, :30], EXPECTED_INPUT_FEATURES, rtol=1e-4, atol=1e-4)

    def test_zero_mean_unit_variance_normalization_trunc_np_longest(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = self._load_datasample(1)
        audio = ((audio - audio.min()) / (audio.max() - audio.min())) * 65535  # Rescale to [0, 65535] to show issue
        audio = feat_extract.zero_mean_unit_var_norm([audio], attention_mask=None)[0]

        self.assertTrue((audio.mean() < 1e-3).all())
        self.assertTrue(((audio.var() - 1).abs() < 1e-3).all())
