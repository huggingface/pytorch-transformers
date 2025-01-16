# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from datasets import Audio, load_dataset

from transformers import UnivNetFeatureExtractor
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, slow
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


class UnivNetFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=1,
        sampling_rate=24000,
        padding_value=0.0,
        do_normalize=True,
        num_mel_bins=100,
        hop_length=256,
        win_length=1024,
        win_function="hann_window",
        filter_length=1024,
        max_length_s=10,
        fmin=0.0,
        fmax=12000,
        mel_floor=1e-9,
        center=False,
        compression_factor=1.0,
        compression_clip_val=1e-5,
        normalize_min=-11.512925148010254,
        normalize_max=2.3143386840820312,
        model_in_channels=64,
        pad_end_length=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)

        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.do_normalize = do_normalize
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.filter_length = filter_length
        self.max_length_s = max_length_s
        self.fmin = fmin
        self.fmax = fmax
        self.mel_floor = mel_floor
        self.center = center
        self.compression_factor = compression_factor
        self.compression_clip_val = compression_clip_val
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max
        self.model_in_channels = model_in_channels
        self.pad_end_length = pad_end_length

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "sampling_rate": self.sampling_rate,
            "padding_value": self.padding_value,
            "do_normalize": self.do_normalize,
            "num_mel_bins": self.num_mel_bins,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "win_function": self.win_function,
            "filter_length": self.filter_length,
            "max_length_s": self.max_length_s,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "mel_floor": self.mel_floor,
            "center": self.center,
            "compression_factor": self.compression_factor,
            "compression_clip_val": self.compression_clip_val,
            "normalize_min": self.normalize_min,
            "normalize_max": self.normalize_max,
            "model_in_channels": self.model_in_channels,
            "pad_end_length": self.pad_end_length,
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


class UnivNetFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = UnivNetFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = UnivNetFeatureExtractionTester(self)

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_feat_extract_from_and_save_pretrained
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

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.test_feat_extract_to_json_file
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

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(
            np_speech_inputs, padding="max_length", max_length=1600, return_tensors="np"
        ).input_features
        self.assertTrue(input_features.ndim == 3)
        # Note: for some reason I get a weird padding error when feature_size > 1
        # self.assertTrue(input_features.shape[-2] == feature_extractor.feature_size)
        # Note: we use the shape convention (batch_size, seq_len, num_mel_bins)
        self.assertTrue(input_features.shape[-1] == feature_extractor.num_mel_bins)

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
        speech_inputs = [
            floats_list((1, x))[0]
            for x in range((feature_extractor.num_max_samples - 100), (feature_extractor.num_max_samples + 500), 200)
        ]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        speech_inputs_truncated = [x[: feature_extractor.num_max_samples] for x in speech_inputs]
        np_speech_inputs_truncated = [np.asarray(speech_input) for speech_input in speech_inputs_truncated]

        encoded_sequences_1 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs_truncated, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_batched_unbatched_consistency(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = floats_list((1, 800))[0]
        np_speech_inputs = np.asarray(speech_inputs)

        # Test unbatched vs batched list
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor([speech_inputs], return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test np.ndarray vs List[np.ndarray]
        encoded_sequences_1 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor([np_speech_inputs], return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

        # Test unbatched np.ndarray vs batched np.ndarray
        encoded_sequences_1 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(
            np.expand_dims(np_speech_inputs, axis=0), return_tensors="np"
        ).input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_generate_noise(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        features = feature_extractor(speech_inputs, return_noise=True)
        input_features = features.input_features
        noise_features = features.noise_sequence

        for spectrogram, noise in zip(input_features, noise_features):
            self.assertEqual(spectrogram.shape[0], noise.shape[0])

    def test_pad_end(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        input_features1 = feature_extractor(speech_inputs, padding=False, pad_end=False).input_features
        input_features2 = feature_extractor(speech_inputs, padding=False, pad_end=True).input_features

        for spectrogram1, spectrogram2 in zip(input_features1, input_features2):
            self.assertEqual(spectrogram1.shape[0] + self.feat_extract_tester.pad_end_length, spectrogram2.shape[0])

    def test_generate_noise_and_pad_end(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        features = feature_extractor(speech_inputs, padding=False, return_noise=True, pad_end=True)
        input_features = features.input_features
        noise_features = features.noise_sequence

        for spectrogram, noise in zip(input_features, noise_features):
            self.assertEqual(spectrogram.shape[0], noise.shape[0])

    @require_torch
    def test_batch_decode(self):
        import torch

        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        input_lengths = list(range(800, 1400, 200))
        pad_samples = feature_extractor.pad_end_length * feature_extractor.hop_length
        output_features = {
            "waveforms": torch.tensor(floats_list((3, max(input_lengths) + pad_samples))),
            "waveform_lengths": torch.tensor(input_lengths),
        }
        waveforms = feature_extractor.batch_decode(**output_features)

        for input_length, waveform in zip(input_lengths, waveforms):
            self.assertTrue(len(waveform.shape) == 1, msg="Individual output waveforms should be 1D")
            self.assertEqual(waveform.shape[0], input_length)

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

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=self.feat_extract_tester.sampling_rate))
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples], [x["sampling_rate"] for x in speech_samples]

    @slow
    @require_torch
    def test_integration(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                -5.0229, -6.1358, -5.8346, -5.4447, -5.6707, -5.8577, -5.0464, -5.0058,
                -5.6015, -5.6410, -5.4325, -5.6116, -5.3700, -5.7956, -5.3196, -5.3274,
                -5.9655, -5.6057, -5.8382, -5.9602, -5.9005, -5.9123, -5.7669, -6.1441,
                -5.5168, -5.1405, -5.3927, -6.0032, -5.5784, -5.3728
            ],
        )
        # fmt: on

        input_speech, sr = self._load_datasamples(1)

        feature_extractor = UnivNetFeatureExtractor()
        input_features = feature_extractor(input_speech, sampling_rate=sr[0], return_tensors="pt").input_features
        self.assertEqual(input_features.shape, (1, 548, 100))

        input_features_mean = torch.mean(input_features)
        input_features_stddev = torch.std(input_features)

        EXPECTED_MEAN = torch.tensor(-6.18862009)
        EXPECTED_STDDEV = torch.tensor(2.80845642)

        torch.testing.assert_close(input_features_mean, EXPECTED_MEAN, rtol=5e-5, atol=5e-5)
        torch.testing.assert_close(input_features_stddev, EXPECTED_STDDEV)
        torch.testing.assert_close(input_features[0, :30, 0], EXPECTED_INPUT_FEATURES, rtol=1e-4, atol=1e-4)
