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
import random
import unittest

import numpy as np
from datasets import load_dataset

from transformers import ClapFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.trainer_utils import set_seed
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
@require_torchaudio
# Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTester with Whisper->Clap
class ClapFeatureExtractionTester:
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


@require_torch
@require_torchaudio
class ClapFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = ClapFeatureExtractor

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest.setUp with Whisper->Clap
    def setUp(self):
        self.feat_extract_tester = ClapFeatureExtractionTester(self)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding="max_length", return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 4)

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

    # Copied from tests.models.whisper.test_feature_extraction_whisper.WhisperFeatureExtractionTest._load_datasamples
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_integration_fusion_short_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    # "repeat"
                    [
                        -20.1049, -19.9764, -20.0731, -19.5055, -27.5018, -22.5761, -26.6071,
                        -29.0091, -26.4659, -26.4236, -28.8808, -31.9190, -32.4848, -34.1186,
                        -34.0340, -32.8803, -30.9895, -37.6238, -38.0347, -40.6263, -36.3496,
                        -42.2533, -32.9132, -27.7068, -29.3704, -30.3208, -22.5972, -27.1494,
                        -30.1975, -31.1005, -29.9372, -27.1917, -25.9806, -30.3489, -33.2380,
                        -31.9062, -36.5498, -32.8721, -30.5629, -27.4674, -22.2232, -22.5653,
                        -16.3868, -17.2713, -25.9738, -30.6256, -34.3766, -31.1292, -27.8950,
                        -27.0588, -25.6206, -23.0712, -26.6050, -28.0112, -32.6847, -34.3396,
                        -34.9738, -35.8463, -39.2324, -37.1188, -33.3705, -28.9230, -28.9112,
                        -28.6578
                    ],
                    [
                        -36.7233, -30.0587, -24.8431, -18.4611, -16.8149, -23.9319, -32.8580,
                        -34.2264, -27.4332, -26.8027, -29.2721, -33.9033, -39.3403, -35.3232,
                        -26.8076, -28.6460, -35.2780, -36.0738, -35.4996, -37.7631, -39.5056,
                        -34.7112, -36.8741, -34.1066, -32.9474, -33.6604, -27.9937, -30.9594,
                        -26.2928, -32.0485, -29.2151, -29.2917, -32.7308, -29.6542, -31.1454,
                        -37.0088, -32.3388, -37.3086, -31.1024, -27.2889, -19.6788, -21.1488,
                        -19.5144, -14.8889, -21.2006, -24.7488, -27.7940, -31.1058, -27.5068,
                        -21.5737, -22.3780, -21.5151, -26.3086, -30.9223, -33.5043, -32.0307,
                        -37.3806, -41.6188, -45.6650, -40.5131, -32.5023, -26.7385, -26.3709,
                        -26.7761
                    ]
                ],
                [
                    # "repeatpad"
                    [
                        -25.7496, -24.9339, -24.1357, -23.1271, -23.7853, -26.1264, -29.1456,
                        -33.2060, -37.8179, -42.4833, -41.9386, -41.2164, -42.3566, -44.2575,
                        -40.0217, -36.6794, -36.6974, -38.7819, -42.0880, -45.5560, -39.9368,
                        -36.3219, -35.5981, -36.6434, -35.1851, -33.0684, -30.0437, -30.2010,
                        -34.3476, -42.1373, -38.8039, -37.3355, -40.4576, -41.0485, -40.6377,
                        -38.2275, -42.7481, -34.6084, -34.7048, -29.5149, -26.3935, -26.8952,
                        -34.1336, -26.2904, -28.2571, -32.5642, -36.7240, -35.5334, -38.2451,
                        -34.8177, -28.9754, -25.1096, -27.9768, -32.3184, -37.0269, -40.5136,
                        -40.8061, -36.4948, -40.3767, -38.9671, -38.3552, -34.1250, -30.9035,
                        -31.6112
                    ],
                    [
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100.
                    ]
                ],
                [
                    # None, same as "repeatpad"
                    [
                        -25.7496, -24.9339, -24.1357, -23.1271, -23.7853, -26.1264, -29.1456,
                        -33.2060, -37.8179, -42.4833, -41.9386, -41.2164, -42.3566, -44.2575,
                        -40.0217, -36.6794, -36.6974, -38.7819, -42.0880, -45.5560, -39.9368,
                        -36.3219, -35.5981, -36.6434, -35.1851, -33.0684, -30.0437, -30.2010,
                        -34.3476, -42.1373, -38.8039, -37.3355, -40.4576, -41.0485, -40.6377,
                        -38.2275, -42.7481, -34.6084, -34.7048, -29.5149, -26.3935, -26.8952,
                        -34.1336, -26.2904, -28.2571, -32.5642, -36.7240, -35.5334, -38.2451,
                        -34.8177, -28.9754, -25.1096, -27.9768, -32.3184, -37.0269, -40.5136,
                        -40.8061, -36.4948, -40.3767, -38.9671, -38.3552, -34.1250, -30.9035,
                        -31.6112
                    ],
                    [
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100.
                    ]
                ],
                [
                    # "pad"
                    [
                        -58.5260, -58.1155, -57.8623, -57.5059, -57.9178, -58.7171, -59.2343,
                        -59.9833, -60.9764, -62.0722, -63.5723, -65.7111, -67.5153, -68.7088,
                        -69.8325, -70.2987, -70.1548, -70.6233, -71.5702, -72.5159, -72.3821,
                        -70.1817, -67.0315, -64.1387, -62.2202, -61.0717, -60.4951, -61.6005,
                        -63.7358, -67.1400, -67.6185, -65.5635, -64.3593, -63.7138, -63.6209,
                        -66.4950, -72.6284, -63.3961, -56.8334, -52.7319, -50.6310, -51.3728,
                        -53.5619, -51.9190, -50.9708, -52.8684, -55.8073, -58.8227, -60.6991,
                        -57.0547, -52.7611, -51.4388, -54.4892, -60.8950, -66.1024, -72.4352,
                        -67.8538, -65.1463, -68.7588, -72.3080, -68.4864, -60.4688, -57.1516,
                        -60.9460
                    ],
                    [
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100.
                    ]
                ]
            ]
        )
        # fmt: on
        MEL_BIN = [[976, 977], [976, 977], [976, 977], [196, 197]]
        input_speech = self._load_datasamples(1)
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, idx_in_mel in zip(
            ["repeat", "repeatpad", None, "pad"], EXPECTED_INPUT_FEATURES, MEL_BIN
        ):
            input_features = feature_extractor(input_speech, return_tensors="pt", padding=padding).input_features
            self.assertEqual(input_features.shape, (1, 4, 1001, 64))

            torch.testing.assert_close(input_features[0, 0, idx_in_mel[0]], EXPECTED_VALUES[0], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(input_features[0, 0, idx_in_mel[1]], EXPECTED_VALUES[1], rtol=1e-4, atol=1e-4)

            self.assertTrue(torch.all(input_features[0, 0] == input_features[0, 1]))
            self.assertTrue(torch.all(input_features[0, 0] == input_features[0, 2]))
            self.assertTrue(torch.all(input_features[0, 0] == input_features[0, 3]))

    def test_integration_rand_trunc_short_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    # "repeat"
                    [
                        -35.0483, -35.7865, -38.2884, -40.0220, -42.5349, -44.9489, -43.2228,
                        -44.6499, -47.6253, -49.6983, -50.2127, -52.5483, -52.2223, -51.9157,
                        -49.4082, -51.2024, -57.0476, -56.2803, -58.1618, -60.7474, -55.0389,
                        -60.9514, -59.3080, -50.4419, -47.8172, -48.7570, -55.2552, -44.5036,
                        -44.1148, -50.8218, -51.0968, -52.9408, -51.1037, -48.9789, -47.5897,
                        -52.0915, -55.4216, -54.1529, -58.0149, -58.0866, -52.7798, -52.6154,
                        -45.9144, -46.2008, -40.7603, -41.1703, -50.2250, -55.4112, -59.4818,
                        -54.5795, -53.5552, -51.3668, -49.8358, -50.3186, -54.0452, -57.6030,
                        -61.1589, -61.6415, -63.2756, -66.5890, -62.8543, -58.0665, -56.7203,
                        -56.7632
                    ],
                    [
                        -47.1320, -37.9961, -34.0076, -36.7109, -47.9057, -48.4924, -43.8371,
                        -44.9728, -48.1689, -52.9141, -57.6077, -52.8520, -44.8502, -45.6764,
                        -51.8389, -56.4284, -54.6972, -53.4889, -55.6077, -58.7149, -60.3760,
                        -54.0136, -56.0730, -55.9870, -54.4017, -53.1094, -53.5640, -50.3064,
                        -49.9520, -49.3239, -48.1668, -53.4852, -50.4561, -50.8688, -55.1970,
                        -51.5538, -53.0260, -59.6933, -54.8183, -59.5895, -55.9589, -50.3761,
                        -44.1282, -44.1463, -43.8540, -39.1168, -45.3893, -49.5542, -53.1505,
                        -55.2870, -50.3921, -46.8511, -47.4444, -49.5633, -56.0034, -59.0815,
                        -59.0018, -63.7589, -69.5745, -71.5789, -64.0498, -56.0558, -54.3475,
                        -54.7004
                    ]
                ],
                [
                    # "repeatpad"
                    [
                        -40.3184, -39.7186, -39.8807, -41.6508, -45.3613, -50.4785, -57.0297,
                        -60.4944, -59.1642, -58.9495, -60.4661, -62.5300, -58.4759, -55.2865,
                        -54.8973, -56.0780, -57.5482, -59.6557, -64.3309, -65.0330, -59.4941,
                        -56.8552, -55.0519, -55.9817, -56.9739, -55.2827, -54.5312, -51.4141,
                        -50.4289, -51.9131, -57.5821, -63.9979, -59.9180, -58.9489, -62.3247,
                        -62.6975, -63.7948, -60.5250, -64.6107, -58.7905, -57.0229, -54.3084,
                        -49.8445, -50.4459, -57.0172, -50.6425, -52.5992, -57.4207, -61.6358,
                        -60.6540, -63.1968, -57.4360, -52.3263, -51.7695, -57.1946, -62.9610,
                        -66.7359, -67.0335, -63.7440, -68.1775, -66.3798, -62.8650, -59.8972,
                        -59.3139
                    ],
                    [
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100.
                    ]
                ],
                [
                    # None, same as "repeatpad"
                    [
                        -40.3184, -39.7186, -39.8807, -41.6508, -45.3613, -50.4785, -57.0297,
                        -60.4944, -59.1642, -58.9495, -60.4661, -62.5300, -58.4759, -55.2865,
                        -54.8973, -56.0780, -57.5482, -59.6557, -64.3309, -65.0330, -59.4941,
                        -56.8552, -55.0519, -55.9817, -56.9739, -55.2827, -54.5312, -51.4141,
                        -50.4289, -51.9131, -57.5821, -63.9979, -59.9180, -58.9489, -62.3247,
                        -62.6975, -63.7948, -60.5250, -64.6107, -58.7905, -57.0229, -54.3084,
                        -49.8445, -50.4459, -57.0172, -50.6425, -52.5992, -57.4207, -61.6358,
                        -60.6540, -63.1968, -57.4360, -52.3263, -51.7695, -57.1946, -62.9610,
                        -66.7359, -67.0335, -63.7440, -68.1775, -66.3798, -62.8650, -59.8972,
                        -59.3139
                    ],
                    [
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100.
                    ]
                ],
                [
                    # "pad"
                    [
                        -73.3190, -73.6349, -74.1451, -74.8539, -75.7476, -76.5438, -78.5540,
                        -80.1339, -81.8911, -83.7560, -85.5387, -86.7466, -88.2072, -88.6090,
                        -88.8243, -89.0784, -89.4364, -89.8179, -91.3146, -92.2833, -91.7221,
                        -90.9440, -88.1315, -86.2425, -84.2281, -82.4893, -81.5993, -81.1328,
                        -81.5759, -83.1068, -85.6525, -88.9520, -88.9187, -87.2703, -86.3052,
                        -85.7188, -85.8802, -87.9996, -95.0464, -88.0133, -80.8561, -76.5597,
                        -74.2816, -74.8109, -77.3615, -76.0719, -75.3426, -77.6428, -80.9663,
                        -84.5275, -84.9907, -80.5205, -77.2851, -78.6259, -84.7740, -91.4535,
                        -98.1894, -94.3872, -92.3735, -97.6807, -98.1501, -91.4344, -85.2842,
                        -88.4338
                    ],
                    [
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100., -100., -100., -100., -100., -100., -100.,
                        -100., -100., -100., -100.
                    ]
                ]
            ]
        )
        # fmt: on
        MEL_BIN = [[976, 977], [976, 977], [976, 977], [196, 197]]
        input_speech = self._load_datasamples(1)
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, idx_in_mel in zip(
            ["repeat", "repeatpad", None, "pad"], EXPECTED_INPUT_FEATURES, MEL_BIN
        ):
            input_features = feature_extractor(
                input_speech, return_tensors="pt", truncation="rand_trunc", padding=padding
            ).input_features
            self.assertEqual(input_features.shape, (1, 1, 1001, 64))
            torch.testing.assert_close(input_features[0, 0, idx_in_mel[0]], EXPECTED_VALUES[0], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(input_features[0, 0, idx_in_mel[1]], EXPECTED_VALUES[1], rtol=1e-4, atol=1e-4)

    def test_integration_fusion_long_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -11.1830, -10.1894, -8.6051, -4.8578, -1.3268, -8.4606, -14.5453,
                     -9.2017, 0.5781, 16.2129, 14.8289, 3.6326, -3.8794, -6.5544,
                     -2.4408, 1.9531, 6.0967, 1.7590, -7.6730, -6.1571, 2.0052,
                     16.6694, 20.6447, 21.2145, 13.4972, 15.9043, 16.8987, 4.1766,
                     11.9428, 21.2372, 12.3016, 4.8604, 6.7241, 1.8543, 4.9235,
                      5.3188, -0.9897, -1.2416, -6.5864, 2.9529, 2.9274, 6.4753,
                     10.2300, 11.2127, 3.4042, -1.0055, -6.0475, -6.7524, -3.9801,
                     -1.4434, 0.4740, -0.1584, -4.5457, -8.5746, -8.8428, -13.1475,
                     -9.6079, -8.5798, -4.1143, -3.7966, -7.1651, -6.1517, -8.0258,
                    -12.1486
                ],
                [
                    -10.2017, -7.9924, -5.9517, -3.9372, -1.9735, -4.3130, 16.1647,
                     25.0592, 23.5532, 14.4974, -7.0778, -10.2262, 6.4782, 20.3454,
                     19.4269, 1.7976, -16.5070, 4.9380, 12.3390, 6.9285, -13.6325,
                     -8.5298, 1.0839, -5.9629, -8.4812, 3.1331, -2.0963, -16.6046,
                    -14.0070, -17.5707, -13.2080, -17.2168, -17.7770, -12.1111, -18.6184,
                    -17.1897, -13.9801, -12.0426, -23.5400, -25.6823, -23.5813, -18.7847,
                    -20.5473, -25.6458, -19.7585, -27.6007, -28.9276, -24.8948, -25.4458,
                    -22.2807, -19.6613, -19.2669, -15.7813, -19.6821, -24.3439, -22.2598,
                    -28.2631, -30.1017, -32.7646, -33.6525, -27.5639, -22.0548, -27.8054,
                    -29.6947
                ],
                [
                    -9.2078, -7.2963, -6.2095, -7.9959, -2.9280, -11.1843, -6.1490,
                    5.0733, 19.2957, 21.4578, 14.6803, -3.3153, -6.3334, -2.3542,
                    6.9509, 15.2965, 14.6620, 5.2075, -0.0873, 1.1919, 18.1986,
                    20.8470, 10.8035, 2.2516, 7.6905, 7.7427, -1.2543, -5.0018,
                    0.9809, -2.1584, -5.4580, -5.4760, -11.8888, -9.0605, -8.4638,
                    -9.9897, -0.0540, -5.1629, 0.0483, -4.1504, -4.8140, -7.8236,
                    -9.0622, -10.1742, -8.9597, -11.5380, -16.5603, -17.1858, -17.5032,
                    -20.9326, -23.9543, -25.2602, -25.3429, -27.4536, -26.8859, -22.7852,
                    -25.8288, -24.8399, -23.8893, -24.2096, -26.5415, -23.7281, -25.6851,
                    -22.3629
                ],
                [
                      1.3448, 2.9883, 4.0366, -0.8019, -10.4191, -10.0883, -4.3812,
                      0.8136, 2.1579, 0.0832, 1.0949, -0.9759, -5.5319, -4.6009,
                     -6.5452, -14.9155, -20.1584, -9.3611, -2.4271, 1.4031, 4.9910,
                      8.6916, 8.6785, 10.1973, 9.9029, 5.3840, 7.5336, 5.2803,
                      2.8144, -0.3138, 2.2216, 5.7328, 7.5574, 7.7402, 1.0681,
                      3.1049, 7.0742, 6.5588, 7.3712, 5.7881, 8.6874, 8.7725,
                      2.8133, -4.5809, -6.1317, -5.1719, -5.0192, -9.0977, -10.9391,
                     -6.0769, 1.6016, -0.8965, -7.2252, -7.8632, -11.4468, -11.7446,
                    -10.7447, -7.0601, -2.7748, -4.1798, -2.8433, -3.1352, 0.8097,
                      6.4212
                ]
            ]
        )
        # fmt: on
        MEL_BIN = 963
        input_speech = torch.cat([torch.tensor(x) for x in self._load_datasamples(5)])
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, block_idx in zip(
            ["repeat", "repeatpad", None, "pad"], EXPECTED_INPUT_FEATURES, [1, 2, 0, 3]
        ):
            set_seed(987654321)
            input_features = feature_extractor(input_speech, return_tensors="pt", padding=padding).input_features
            self.assertEqual(input_features.shape, (1, 4, 1001, 64))
            torch.testing.assert_close(input_features[0, block_idx, MEL_BIN], EXPECTED_VALUES, rtol=1e-3, atol=1e-3)

    def test_integration_rand_trunc_long_input(self):
        # fmt: off
        EXPECTED_INPUT_FEATURES = torch.tensor(
            [
                [
                    -35.4022, -32.7555, -31.2004, -32.7764, -42.5770, -41.6339, -43.1630,
                    -44.5080, -44.3029, -48.9628, -39.5022, -39.2105, -43.1350, -43.2195,
                    -48.4894, -52.2344, -57.6891, -52.2228, -45.5155, -44.2893, -43.4697,
                    -46.6702, -43.7490, -40.4819, -42.7275, -46.3434, -46.8412, -41.2003,
                    -43.1681, -46.2948, -46.1925, -47.8333, -45.6812, -44.9182, -41.7786,
                    -43.3809, -44.3199, -42.8814, -45.4771, -46.7114, -46.9746, -42.7090,
                    -41.6057, -38.3965, -40.1980, -41.0263, -34.1256, -28.3289, -29.0201,
                    -30.4453, -29.5561, -30.1734, -25.9406, -19.0897, -15.8452, -20.1351,
                    -23.6515, -23.1194, -17.1845, -19.4399, -23.6527, -22.8768, -20.7279,
                    -22.7864
                ],
                [
                    -35.7719, -27.2566, -23.6964, -27.5521, 0.2510, 7.4391, 1.3917,
                    -13.3417, -28.1758, -17.0856, -5.7723, -0.8000, -7.8832, -15.5548,
                    -30.5935, -24.7571, -13.7009, -10.3432, -21.2464, -24.8118, -19.4080,
                    -14.9779, -11.7991, -18.4485, -20.1982, -17.3652, -20.6328, -28.2967,
                    -25.7819, -21.8962, -28.5083, -29.5719, -30.2120, -35.7033, -31.8218,
                    -34.0408, -37.7744, -33.9653, -31.3009, -30.9063, -28.6153, -32.2202,
                    -28.5456, -28.8579, -32.5170, -37.9152, -43.0052, -46.4849, -44.0786,
                    -39.1933, -33.2757, -31.6313, -42.6386, -52.3679, -53.5785, -55.6444,
                    -47.0050, -47.6459, -56.6361, -60.6781, -61.5244, -55.8272, -60.4832,
                    -58.1897
                ],
                [
                    -38.2686, -36.6285, -32.5835, -35.1693, -37.7938, -37.4035, -35.3132,
                    -35.6083, -36.3609, -40.9472, -36.7846, -36.1544, -38.9076, -39.3618,
                    -35.4953, -34.2809, -39.9466, -39.7433, -34.8347, -37.5674, -41.5689,
                    -38.9161, -34.3947, -30.2924, -30.4841, -34.5831, -28.9261, -24.8849,
                    -31.2324, -27.1622, -27.2107, -25.9385, -30.1691, -30.9223, -23.9495,
                    -25.6047, -26.7119, -28.5523, -27.7481, -32.8427, -35.4650, -31.0399,
                    -31.2073, -30.5163, -22.9819, -20.8892, -19.2510, -24.7905, -28.9426,
                    -28.1998, -26.7386, -25.0140, -27.9223, -32.9913, -33.1864, -34.9742,
                    -38.5995, -39.6990, -29.3203, -22.4697, -25.6415, -33.5608, -33.0945,
                    -27.1716
                ],
                [
                    -33.2015, -28.7741, -21.9457, -23.4888, -32.1072, -8.6307, 3.2724,
                      5.9157, -0.9221, -30.1814, -31.0015, -27.4508, -27.0477, -9.5342,
                      0.3221, 0.6511, -7.1596, -25.9707, -32.8924, -32.2300, -13.8974,
                     -0.4895, 0.9168, -10.7663, -27.1176, -35.0829, -11.6859, -4.8855,
                    -11.8898, -26.6167, -5.6192, -3.8443, -19.7947, -14.4101, -8.6236,
                    -21.2458, -21.0801, -17.9136, -24.4663, -18.6333, -24.8085, -15.5854,
                    -15.4344, -11.5046, -22.3625, -27.3387, -32.4353, -30.9670, -31.3789,
                    -35.4044, -34.4591, -25.2433, -28.0773, -33.8736, -33.0224, -33.3155,
                    -38.5302, -39.2741, -36.6395, -34.7729, -32.4483, -42.4001, -49.2857,
                    -39.1682
                ]
            ]
        )
        # fmt: on
        MEL_BIN = 963
        SEEDS = [987654321, 1234, 666, 5555]
        input_speech = torch.cat([torch.tensor(x) for x in self._load_datasamples(5)])
        feature_extractor = ClapFeatureExtractor()
        for padding, EXPECTED_VALUES, seed in zip(
            ["repeat", "repeatpad", None, "pad"], EXPECTED_INPUT_FEATURES, SEEDS
        ):
            set_seed(seed)
            input_features = feature_extractor(
                input_speech, return_tensors="pt", truncation="rand_trunc", padding=padding
            ).input_features
            self.assertEqual(input_features.shape, (1, 1, 1001, 64))
            torch.testing.assert_close(input_features[0, 0, MEL_BIN], EXPECTED_VALUES, rtol=1e-4, atol=1e-4)
