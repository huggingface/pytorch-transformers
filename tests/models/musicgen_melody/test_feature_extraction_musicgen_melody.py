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


import itertools
import math
import os
import random
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
    require_torchaudio,
)
from transformers.utils.import_utils import is_torchaudio_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_torchaudio_available():
    import torch

    from transformers import MusicgenMelodyFeatureExtractor


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


# Copied from tests.models.musicgen.test_modeling_musicgen.get_bip_bip
def get_bip_bip(bip_duration=0.125, duration=0.5, sample_rate=32000):
    """Produces a series of 'bip bip' sounds at a given frequency."""
    timesteps = np.arange(int(duration * sample_rate)) / sample_rate
    wav = np.cos(2 * math.pi * 440 * timesteps)
    time_period = (timesteps % (2 * bip_duration)) / (2 * bip_duration)
    envelope = time_period >= 0.5
    return wav * envelope


@require_torch
@require_torchaudio
class MusicgenMelodyFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=12,
        padding_value=0.0,
        sampling_rate=4_000,
        return_attention_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.feature_size = feature_size
        self.num_chroma = feature_size

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
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


@require_torchaudio
@require_torch
class MusicgenMelodyFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = MusicgenMelodyFeatureExtractor if is_torchaudio_available() else None

    def setUp(self):
        self.feat_extract_tester = MusicgenMelodyFeatureExtractionTester(self)

    # Copied from tests.models.seamless_m4t.test_feature_extraction_seamless_m4t.SeamlessM4TFeatureExtractionTest.test_feat_extract_from_and_save_pretrained
    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertDictEqual(dict_first, dict_second)

    # Copied from tests.models.seamless_m4t.test_feature_extraction_seamless_m4t.SeamlessM4TFeatureExtractionTest.test_feat_extract_to_json_file
    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding=True, return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[0] == 3)
        # Ignore copy
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size)

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

    @require_torchaudio
    def test_call_from_demucs(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())

        # (batch_size, num_stems, channel_size, audio_length)
        inputs = torch.rand([4, 5, 2, 44000])

        # Test feature size
        input_features = feature_extractor(inputs, padding=True, return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[0] == 4)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size)

        # Test single input
        encoded_sequences_1 = feature_extractor(inputs[[0]], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1[0], input_features[0], atol=1e-3))

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_double_precision_pad with input_features->input_features
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

    def test_integration(self):
        EXPECTED_INPUT_FEATURES = torch.zeros([2, 8, 12])
        EXPECTED_INPUT_FEATURES[0, :6, 9] = 1
        EXPECTED_INPUT_FEATURES[0, 6:, 0] = 1
        EXPECTED_INPUT_FEATURES[1, :, 9] = 1

        input_speech = [get_bip_bip(duration=0.5), get_bip_bip(duration=1.0)]
        feature_extractor = MusicgenMelodyFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").input_features

        self.assertEqual(input_features.shape, (2, 8, 12))
        self.assertTrue((input_features == EXPECTED_INPUT_FEATURES).all())
