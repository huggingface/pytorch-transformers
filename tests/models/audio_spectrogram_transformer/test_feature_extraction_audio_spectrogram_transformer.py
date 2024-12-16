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

from transformers import ASTFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_torchaudio
from transformers.utils.import_utils import is_torch_available

from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


global_rng = random.Random()

if is_torch_available():
    import torch


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


class ASTFeatureExtractionTester:
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


@require_torch
@require_torchaudio
class ASTFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = ASTFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = ASTFeatureExtractionTester(self)

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
        encoded_sequences_1 = feat_extract(speech_inputs, padding=True, return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs, padding=True, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors="np").input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

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

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @require_torch
    def test_integration(self):
        # fmt: off
        EXPECTED_INPUT_VALUES = torch.tensor(
            [-0.9894, -1.2776, -0.9066, -1.2776, -0.9349, -1.2609, -1.0386, -1.2776,
             -1.1561, -1.2776, -1.2052, -1.2723, -1.2190, -1.2132, -1.2776, -1.1133,
             -1.1953, -1.1343, -1.1584, -1.2203, -1.1770, -1.2474, -1.2381, -1.1936,
             -0.9270, -0.8317, -0.8049, -0.7706, -0.7565, -0.7869]
        )
        # fmt: on

        input_speech = self._load_datasamples(1)
        feature_extractor = ASTFeatureExtractor()
        input_values = feature_extractor(input_speech, return_tensors="pt").input_values
        self.assertEqual(input_values.shape, (1, 1024, 128))
        self.assertTrue(torch.allclose(input_values[0, 0, :30], EXPECTED_INPUT_VALUES, atol=1e-4))

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


# exact same tests than before, except that we simulate that torchaudio is not available
@require_torch
@unittest.mock.patch(
    "transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer.is_speech_available",
    lambda: False,
)
class ASTFeatureExtractionWithoutTorchaudioTest(ASTFeatureExtractionTest):
    def test_using_audio_utils(self):
        # Tests that it uses audio_utils instead of torchaudio
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())

        self.assertTrue(hasattr(feat_extract, "window"))
        self.assertTrue(hasattr(feat_extract, "mel_filters"))

        from transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer import (
            is_speech_available,
        )

        self.assertFalse(is_speech_available())
