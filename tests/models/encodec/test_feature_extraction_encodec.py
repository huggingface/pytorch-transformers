# coding=utf-8
# Copyright 2021-2023 HuggingFace Inc.
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
"""Tests for the EnCodec feature extractor."""

import itertools
import random
import unittest

import numpy as np

from transformers import EncodecFeatureExtractor
from transformers.testing_utils import require_torch
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
class EnCodecFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        padding_value=0.0,
        sampling_rate=24000,
        return_attention_mask=True,
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

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            audio_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            audio_inputs = [
                _flatten(floats_list((x, self.feature_size)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            audio_inputs = [np.asarray(x) for x in audio_inputs]

        return audio_inputs


@require_torch
class EnCodecFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = EncodecFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = EnCodecFeatureExtractionTester(self)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        audio_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_audio_inputs = [np.asarray(audio_input) for audio_input in audio_inputs]

        # Test not batched input
        encoded_sequences_1 = feat_extract(audio_inputs[0], return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_audio_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feat_extract(audio_inputs, padding=True, return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_audio_inputs, padding=True, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_double_precision_pad(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_audio_inputs = np.random.rand(100).astype(np.float64)
        py_audio_inputs = np_audio_inputs.tolist()

        for inputs in [py_audio_inputs, np_audio_inputs]:
            np_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_values.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_values": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_values.dtype == torch.float32)

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        audio_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in audio_samples]

    def test_integration(self):
        # fmt: off
        EXPECTED_INPUT_VALUES = torch.tensor(
            [2.3804e-03, 2.0752e-03, 1.9836e-03, 2.1057e-03, 1.6174e-03,
             3.0518e-04, 9.1553e-05, 3.3569e-04, 9.7656e-04, 1.8311e-03,
             2.0142e-03, 2.1057e-03, 1.7395e-03, 4.5776e-04, -3.9673e-04,
             4.5776e-04, 1.0071e-03, 9.1553e-05, 4.8828e-04, 1.1597e-03,
             7.3242e-04, 9.4604e-04, 1.8005e-03, 1.8311e-03, 8.8501e-04,
             4.2725e-04, 4.8828e-04, 7.3242e-04, 1.0986e-03, 2.1057e-03]
        )
        # fmt: on
        input_audio = self._load_datasamples(1)
        feature_extractor = EncodecFeatureExtractor()
        input_values = feature_extractor(input_audio, return_tensors="pt").input_values
        self.assertEqual(input_values.shape, (1, 1, 93680))
        torch.testing.assert_close(input_values[0, 0, :30], EXPECTED_INPUT_VALUES, rtol=1e-6, atol=1e-6)

    def test_integration_stereo(self):
        # fmt: off
        EXPECTED_INPUT_VALUES = torch.tensor(
            [2.3804e-03, 2.0752e-03, 1.9836e-03, 2.1057e-03, 1.6174e-03,
             3.0518e-04, 9.1553e-05, 3.3569e-04, 9.7656e-04, 1.8311e-03,
             2.0142e-03, 2.1057e-03, 1.7395e-03, 4.5776e-04, -3.9673e-04,
             4.5776e-04, 1.0071e-03, 9.1553e-05, 4.8828e-04, 1.1597e-03,
             7.3242e-04, 9.4604e-04, 1.8005e-03, 1.8311e-03, 8.8501e-04,
             4.2725e-04, 4.8828e-04, 7.3242e-04, 1.0986e-03, 2.1057e-03]
        )
        # fmt: on
        input_audio = self._load_datasamples(1)
        input_audio = [np.tile(input_audio[0][None], reps=(2, 1))]
        input_audio[0][1] *= 0.5
        feature_extractor = EncodecFeatureExtractor(feature_size=2)
        input_values = feature_extractor(input_audio, return_tensors="pt").input_values
        self.assertEqual(input_values.shape, (1, 2, 93680))
        torch.testing.assert_close(input_values[0, 0, :30], EXPECTED_INPUT_VALUES, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(input_values[0, 1, :30], EXPECTED_INPUT_VALUES * 0.5, rtol=1e-6, atol=1e-6)

    def test_truncation_and_padding(self):
        input_audio = self._load_datasamples(2)
        # would be easier if the stride was like
        feature_extractor = EncodecFeatureExtractor(feature_size=1, chunk_length_s=1, overlap=0.01)

        # pad and trunc raise an error ?
        with self.assertRaisesRegex(
            ValueError,
            "^Both padding and truncation were set. Make sure you only set one.$",
        ):
            truncated_outputs = feature_extractor(
                input_audio, padding="max_length", truncation=True, return_tensors="pt"
            ).input_values

        # truncate to chunk
        truncated_outputs = feature_extractor(input_audio, truncation=True, return_tensors="pt").input_values
        self.assertEqual(truncated_outputs.shape, (2, 1, 71520))  # 2 chunks

        # force truncate to max_length
        truncated_outputs = feature_extractor(
            input_audio, truncation=True, max_length=48000, return_tensors="pt"
        ).input_values
        self.assertEqual(truncated_outputs.shape, (2, 1, 48000))

        # pad to chunk
        padded_outputs = feature_extractor(input_audio, padding=True, return_tensors="pt").input_values
        self.assertEqual(padded_outputs.shape, (2, 1, 95280))

        # pad to chunk
        truncated_outputs = feature_extractor(input_audio, return_tensors="pt").input_values
        self.assertEqual(truncated_outputs.shape, (2, 1, 95280))

        # force pad to max length
        truncated_outputs = feature_extractor(
            input_audio, padding="max_length", max_length=100000, return_tensors="pt"
        ).input_values
        self.assertEqual(truncated_outputs.shape, (2, 1, 100000))

        # force no pad
        with self.assertRaisesRegex(
            ValueError,
            "^Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.$",
        ):
            truncated_outputs = feature_extractor(input_audio, padding=False, return_tensors="pt").input_values

        truncated_outputs = feature_extractor(input_audio[0], padding=False, return_tensors="pt").input_values
        self.assertEqual(truncated_outputs.shape, (1, 1, 93680))

        # no pad if no chunk_length_s
        feature_extractor.chunk_length_s = None
        with self.assertRaisesRegex(
            ValueError,
            "^Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.$",
        ):
            truncated_outputs = feature_extractor(input_audio, padding=False, return_tensors="pt").input_values

        truncated_outputs = feature_extractor(input_audio[0], padding=False, return_tensors="pt").input_values
        self.assertEqual(truncated_outputs.shape, (1, 1, 93680))

        # no pad if no overlap
        feature_extractor.chunk_length_s = 2
        feature_extractor.overlap = None
        with self.assertRaisesRegex(
            ValueError,
            "^Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.$",
        ):
            truncated_outputs = feature_extractor(input_audio, padding=False, return_tensors="pt").input_values

        truncated_outputs = feature_extractor(input_audio[0], padding=False, return_tensors="pt").input_values
        self.assertEqual(truncated_outputs.shape, (1, 1, 93680))
