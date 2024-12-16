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


import itertools
import random
import unittest

import numpy as np

from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.testing_utils import require_torch, slow

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


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


class Wav2Vec2FeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=16000,
        return_attention_mask=True,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                _flatten(floats_list((x, self.feature_size)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]

        return speech_inputs


class Wav2Vec2FeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Wav2Vec2FeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Wav2Vec2FeatureExtractionTester(self)

    def _check_zero_mean_unit_variance(self, input_vector):
        self.assertTrue(np.all(np.mean(input_vector, axis=0) < 1e-3))
        self.assertTrue(np.all(np.abs(np.var(input_vector, axis=0) - 1) < 1e-3))

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_sequences_1 = feat_extract(speech_inputs[0], return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_zero_mean_unit_variance_normalization_np(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        paddings = ["longest", "max_length", "do_not_pad"]
        max_lengths = [None, 1600, None]
        for max_length, padding in zip(max_lengths, paddings):
            processed = feat_extract(speech_inputs, padding=padding, max_length=max_length, return_tensors="np")
            input_values = processed.input_values

            self._check_zero_mean_unit_variance(input_values[0][:800])
            self.assertTrue(input_values[0][800:].sum() < 1e-6)
            self._check_zero_mean_unit_variance(input_values[1][:1000])
            self.assertTrue(input_values[0][1000:].sum() < 1e-6)
            self._check_zero_mean_unit_variance(input_values[2][:1200])

    def test_zero_mean_unit_variance_normalization(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        lengths = range(800, 1400, 200)
        speech_inputs = [floats_list((1, x))[0] for x in lengths]

        paddings = ["longest", "max_length", "do_not_pad"]
        max_lengths = [None, 1600, None]

        for max_length, padding in zip(max_lengths, paddings):
            processed = feat_extract(speech_inputs, max_length=max_length, padding=padding)
            input_values = processed.input_values

            self._check_zero_mean_unit_variance(input_values[0][:800])
            self._check_zero_mean_unit_variance(input_values[1][:1000])
            self._check_zero_mean_unit_variance(input_values[2][:1200])

    def test_zero_mean_unit_variance_normalization_trunc_np_max_length(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = feat_extract(
            speech_inputs, truncation=True, max_length=1000, padding="max_length", return_tensors="np"
        )
        input_values = processed.input_values

        self._check_zero_mean_unit_variance(input_values[0, :800])
        self._check_zero_mean_unit_variance(input_values[1])
        self._check_zero_mean_unit_variance(input_values[2])

    def test_zero_mean_unit_variance_normalization_trunc_np_longest(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = feat_extract(
            speech_inputs, truncation=True, max_length=1000, padding="longest", return_tensors="np"
        )
        input_values = processed.input_values

        self._check_zero_mean_unit_variance(input_values[0, :800])
        self._check_zero_mean_unit_variance(input_values[1, :1000])
        self._check_zero_mean_unit_variance(input_values[2])

        # make sure that if max_length < longest -> then pad to max_length
        self.assertTrue(input_values.shape == (3, 1000))

        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = feat_extract(
            speech_inputs, truncation=True, max_length=2000, padding="longest", return_tensors="np"
        )
        input_values = processed.input_values

        self._check_zero_mean_unit_variance(input_values[0, :800])
        self._check_zero_mean_unit_variance(input_values[1, :1000])
        self._check_zero_mean_unit_variance(input_values[2])

        # make sure that if max_length > longest -> then pad to longest
        self.assertTrue(input_values.shape == (3, 1200))

    @require_torch
    def test_double_precision_pad(self):
        import torch

        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_values.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_values.dtype == torch.float32)

    @slow
    @require_torch
    def test_pretrained_checkpoints_are_set_correctly(self):
        # this test makes sure that models that are using
        # group norm don't have their feature extractor return the
        # attention_mask
        model_id = "facebook/wav2vec2-base-960h"
        config = Wav2Vec2Config.from_pretrained(model_id)
        feat_extract = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        # only "layer" feature extraction norm should make use of
        # attention_mask
        self.assertEqual(feat_extract.return_attention_mask, config.feat_extract_norm == "layer")
