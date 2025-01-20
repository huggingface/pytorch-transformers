# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest

import numpy as np

from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from transformers.models.whisper.tokenization_whisper import _combine_tokens_into_words, _find_longest_common_sequence
from transformers.testing_utils import require_flax, require_tf, require_torch, slow

from ...test_tokenization_common import TokenizerTesterMixin


ES_CODE = 50262
EN_CODE = 50259
END_OF_TRANSCRIPT = 50257
START_OF_TRANSCRIPT = 50258
TRANSLATE = 50358
TRANSCRIBE = 50359
NOTIMESTAMPS = 50363


class WhisperTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "openai/whisper-tiny"
    tokenizer_class = WhisperTokenizer
    rust_tokenizer_class = WhisperTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = False
    test_seq2seq = False

    def setUp(self):
        super().setUp()
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        tokenizer.pad_token_id = 50256
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "Where"
        token_id = 14436

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "!")
        self.assertEqual(vocab_keys[1], '"')
        self.assertEqual(vocab_keys[-1], "<|30.00|>")
        self.assertEqual(len(vocab_keys), 51865)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 50258)

    def test_full_tokenizer(self):
        tokenizer = WhisperTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["This", "Ġis", "Ġa", "Ġtest"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [5723, 307, 257, 1500],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            ["I", "Ġwas", "Ġborn", "Ġin", "Ġ9", "2000", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", "."],  # fmt: skip
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [40, 390, 4232, 294, 1722, 25743, 11, 293, 341, 307, 16720, 526, 13])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["I", "Ġwas", "Ġborn", "Ġin", "Ġ9", "2000", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", "."],  # fmt: skip
        )

    @unittest.skip
    def test_tokenizer_slow_store_full_signature(self):
        pass

    @unittest.skip
    def test_tokenizer_fast_store_full_signature(self):
        pass

    @unittest.skip
    def test_special_tokens_initialization(self):
        # Whisper relies on specific additional special tokens, so we skip this
        # general test. In particular, this test loads fast tokenizer from slow
        # tokenizer, and the conversion uses prefix_tokens, where we reference
        # additional special tokens by specific indices, hence overriding the
        # list with less tokens leads to out of index error
        pass

    @slow
    def test_tokenizer_integration(self):
        expected_encoding = {'input_ids': [[50257, 50362, 41762, 364, 357, 36234, 1900, 355, 12972, 13165, 354, 12, 35636, 364, 290, 12972, 13165, 354, 12, 5310, 13363, 12, 4835, 8, 3769, 2276, 12, 29983, 45619, 357, 13246, 51, 11, 402, 11571, 12, 17, 11, 5564, 13246, 38586, 11, 16276, 44, 11, 4307, 346, 33, 861, 11, 16276, 7934, 23029, 329, 12068, 15417, 28491, 357, 32572, 52, 8, 290, 12068, 15417, 16588, 357, 32572, 38, 8, 351, 625, 3933, 10, 2181, 13363, 4981, 287, 1802, 10, 8950, 290, 2769, 48817, 1799, 1022, 449, 897, 11, 9485, 15884, 354, 290, 309, 22854, 37535, 13, 50256], [50257, 50362, 13246, 51, 318, 3562, 284, 662, 12, 27432, 2769, 8406, 4154, 282, 24612, 422, 9642, 9608, 276, 2420, 416, 26913, 21143, 319, 1111, 1364, 290, 826, 4732, 287, 477, 11685, 13, 50256], [50257, 50362, 464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding, model_name="openai/whisper-tiny.en", padding=False
        )

    def test_output_offsets(self):
        tokenizer = self.get_tokenizer()
        previous_sequence = [51492, 406, 3163, 1953, 466, 13, 51612, 51612]
        self.assertEqual(
            tokenizer.decode(previous_sequence, output_offsets=True),
            {
                "text": " not worth thinking about.",
                "offsets": [{"text": " not worth thinking about.", "timestamp": (22.56, 24.96)}],
            },
        )

        # Merge when the previous sequence is a suffix of the next sequence
        next_sequences_1 = [50364, 295, 6177, 3391, 11, 19817, 3337, 507, 307, 406, 3163, 1953, 466, 13, 50614, 50614, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50834, 50257]  # fmt: skip
        self.assertEqual(
            tokenizer.decode(next_sequences_1, output_offsets=True),
            {
                "text": (
                    " of spectators, retrievality is not worth thinking about. His instant panic was followed by a"
                    " small, sharp blow high on his chest.<|endoftext|>"
                ),
                "offsets": [
                    {"text": " of spectators, retrievality is not worth thinking about.", "timestamp": (0.0, 5.0)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (5.0, 9.4),
                    },
                ],
            },
        )

    def test_find_longest_common_subsequence(self):
        previous_sequence = [1, 2, 3]
        next_sequence = [2, 3, 4, 5]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5])

        # Now previous is larger than next.
        # We merge what we can and remove the extra right side of the left sequence
        previous_sequence = [1, 2, 3, 4, 5, 6, 7]
        next_sequence = [2, 3, 4, 5]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5])

        # Nothing in common
        previous_sequence = [1, 2, 3]
        next_sequence = [4, 5, 6]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5, 6])

        # Some errors in the overlap.
        # We take from previous on the left, from the next on the right of the overlap
        previous_sequence = [1, 2, 3, 4, 99]
        next_sequence = [2, 98, 4, 5, 6]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5, 6])

        # We take from previous on the left, from the next on the right of the overlap
        previous_sequence = [1, 2, 99, 4, 5]
        next_sequence = [2, 3, 4, 98, 6]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 99, 4, 98, 6])

        # This works on 3 sequences
        seq1 = [1, 2, 3]
        seq2 = [2, 3, 4]
        seq3 = [3, 4, 5]
        merge = _find_longest_common_sequence([seq1, seq2, seq3])
        self.assertEqual(merge, [1, 2, 3, 4, 5])

        # This works on 3 sequences with errors
        seq1 = [1, 2, 3, 98, 5]
        seq2 = [2, 99, 4, 5, 6, 7]
        seq3 = [4, 97, 6, 7, 8]
        merge = _find_longest_common_sequence([seq1, seq2, seq3])
        self.assertEqual(merge, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_skip_special_tokens_skips_prompt_ids(self):
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()
        # fmt: off
        encoded_input = [
            50361, 2221, 13, 2326, 388, 391, 50258, 50259, 50359,
            50363, 1282, 264, 2674, 9156, 295, 1523, 11, 2221, 13,
            2326, 388, 391, 13657, 365, 2681, 21296, 17711, 13, 50257,
        ]
        # fmt: on
        expected_with_special_tokens = "<|startofprev|> Mr. Quilter<|startoftranscript|><|en|><|transcribe|><|notimestamps|> On the general principles of art, Mr. Quilter writes with equal lucidity.<|endoftext|>"
        expected_without_special_tokens = " On the general principles of art, Mr. Quilter writes with equal lucidity."
        self.assertEqual(tokenizer.decode(encoded_input, skip_special_tokens=False), expected_with_special_tokens)
        self.assertEqual(tokenizer.decode(encoded_input, skip_special_tokens=True), expected_without_special_tokens)
        self.assertEqual(rust_tokenizer.decode(encoded_input, skip_special_tokens=False), expected_with_special_tokens)
        self.assertEqual(
            rust_tokenizer.decode(encoded_input, skip_special_tokens=True), expected_without_special_tokens
        )

    def test_skip_special_tokens_with_timestamps(self):
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        # fmt: off
        encoded_input = [
            50258, 50363, 50364, 634, 575, 12525, 22618, 1968, 6144,
            35617, 20084, 1756, 311, 589, 307, 534, 10281, 934,
            439, 293, 50676, 50676, 393, 4411, 294, 309, 457,
            707, 295, 33301, 286, 392, 6628, 13, 50836, 50257,
        ]
        # fmt: on

        expected_with_special_tokens = "<|startoftranscript|><|notimestamps|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all and<|6.24|><|6.24|> can discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>"
        expected_without_special_tokens = "<|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all and<|6.24|><|6.24|> can discover in it but little of rocky Ithaca.<|9.44|>"
        self.assertEqual(
            tokenizer.decode(encoded_input, decode_with_timestamps=True, skip_special_tokens=False),
            expected_with_special_tokens,
        )
        self.assertEqual(
            tokenizer.decode(encoded_input, decode_with_timestamps=True, skip_special_tokens=True),
            expected_without_special_tokens,
        )
        self.assertEqual(
            rust_tokenizer.decode(encoded_input, decode_with_timestamps=True, skip_special_tokens=False),
            expected_with_special_tokens,
        )
        self.assertEqual(
            rust_tokenizer.decode(encoded_input, decode_with_timestamps=True, skip_special_tokens=True),
            expected_without_special_tokens,
        )

    def test_fast_tokenizer_get_prompt_ids(self):
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        prompt = "This is test prompt text."
        tokenizer_prompt_ids = tokenizer.get_prompt_ids(prompt)
        fast_tokenizer_prompt_ids = rust_tokenizer.get_prompt_ids(prompt)

        self.assertListEqual(tokenizer_prompt_ids.tolist(), fast_tokenizer_prompt_ids.tolist())

    def test_tokenizer_decode_prompt(self):
        prompt_text = "What does the fox say?"
        input_text = "Hatee hatee hatee ho"

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        # encode prompt and input text using tokenizer
        prompt_ids = tokenizer.get_prompt_ids(prompt_text, return_tensors="np")
        input_ids = tokenizer(input_text, return_tensors="np").input_ids[0]
        input_ids = np.hstack([prompt_ids, input_ids])

        # encode using fast tokenizer
        rust_prompt_ids = rust_tokenizer.get_prompt_ids(prompt_text, return_tensors="np")
        rust_input_ids = rust_tokenizer(input_text, return_tensors="np").input_ids[0]
        rust_input_ids = np.hstack([rust_prompt_ids, rust_input_ids])

        # check with prompt in output
        pred_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        rust_pred_text = rust_tokenizer.decode(rust_input_ids, skip_special_tokens=False)

        # check correctness for both tokenizers
        expected_text = f"<|startofprev|> {prompt_text}<|startoftranscript|><|notimestamps|>{input_text}<|endoftext|>"
        self.assertEqual(pred_text.strip(), expected_text)
        self.assertEqual(rust_pred_text.strip(), expected_text)

        # check stripping prompt from output
        pred_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        rust_pred_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        self.assertEqual(pred_text.strip(), input_text)
        self.assertEqual(rust_pred_text.strip(), input_text)

    def test_combine_tokens_into_words(self):
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        # 'whatever "whatever" said someone, clever!?'
        encoded_input = [1363, 7969, 503, 1363, 7969, 1, 848, 1580, 11, 13494, 7323]
        expected_words = ["whatever", ' "whatever"', " said", " someone,", " clever!?"]
        expected_tokens = [[1363, 7969], [503, 1363, 7969, 1], [848], [1580, 11], [13494, 7323]]
        expected_indices = [[0, 1], [2, 3, 4, 5], [6], [7, 8], [9, 10]]
        output = _combine_tokens_into_words(tokenizer, encoded_input)
        self.assertEqual(expected_words, output[0])
        self.assertEqual(expected_tokens, output[1])
        self.assertEqual(expected_indices, output[2])
        output_rust = _combine_tokens_into_words(rust_tokenizer, encoded_input)
        self.assertEqual(expected_words, output_rust[0])
        self.assertEqual(expected_tokens, output_rust[1])
        self.assertEqual(expected_indices, output_rust[2])

    def test_basic_normalizer(self):
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        input_str = "Hola güey!"
        expected_output_normalize = "hola güey "
        expected_output_diacritics = "hola guey "

        # tokenizer tests
        encoded_input = tokenizer(input_str).input_ids
        decoded_output = tokenizer.decode(encoded_input, skip_special_tokens=True, basic_normalize=False)
        self.assertEqual(decoded_output, input_str)

        decoded_output_normalize = tokenizer.decode(encoded_input, skip_special_tokens=True, basic_normalize=True)
        self.assertEqual(decoded_output_normalize, expected_output_normalize)

        decoded_output_diacritics = tokenizer.decode(
            encoded_input, skip_special_tokens=True, basic_normalize=True, remove_diacritics=True
        )
        self.assertEqual(decoded_output_diacritics, expected_output_diacritics)

        # fast tokenizer tests
        encoded_input = rust_tokenizer(input_str).input_ids
        decoded_output = rust_tokenizer.decode(encoded_input, skip_special_tokens=True, basic_normalize=False)
        self.assertEqual(decoded_output, input_str)

        decoded_output_normalize = rust_tokenizer.decode(encoded_input, skip_special_tokens=True, basic_normalize=True)
        self.assertEqual(decoded_output_normalize, expected_output_normalize)

        decoded_output_diacritics = rust_tokenizer.decode(
            encoded_input, skip_special_tokens=True, basic_normalize=True, remove_diacritics=True
        )
        self.assertEqual(decoded_output_diacritics, expected_output_diacritics)

    def test_decode_asr_with_word_level_timestamps(self):
        # fmt: off
        model_outputs = [
            {
                'stride': [10, 0, 5],
                'tokens': np.array([[ 50257, 50362, 3363, 11, 345, 460, 0, 2329, 466, 340, 0, 50256 ]]),
                'token_timestamps': np.array([[ 0, 0, 5.18, 5.56, 5.56, 5.84, 6.36, 7.12, 7.54, 7.82, 8.16, 9.48 ]])
            },
            {
                'stride': [10, 5, 0],
                'tokens': np.array([[ 50257, 50362, 2329, 466, 340, 0, 3363, 345, 460, 0, 2329, 466, 340, 50256 ]]),
                'token_timestamps': np.array([[ 0, 0, 0, 2.44, 4.3, 5.04, 5.06, 5.56, 5.8, 6.32, 7.12, 7.56, 7.8, 8.72 ]])
            }
        ]
        # fmt: on

        tokenizer = WhisperTokenizer.from_pretrained("onnx-community/whisper-tiny.en_timestamped")
        result = tokenizer._decode_asr(
            model_outputs, return_timestamps="word", return_language=False, time_precision=0.02
        )

        EXPECTED_OUTPUT = (
            " Yes, you can! Just do it",
            {
                "chunks": [
                    {"text": " Yes,", "timestamp": (5.18, 5.56)},
                    {"text": " you", "timestamp": (5.56, 5.84)},
                    {"text": " can!", "timestamp": (5.84, 7.12)},
                    {"text": " Just", "timestamp": (7.12, 7.56)},
                    {"text": " do", "timestamp": (7.56, 7.8)},
                    {"text": " it", "timestamp": (7.8, 8.72)},
                ]
            },
        )
        self.assertEqual(result, EXPECTED_OUTPUT)

        # fmt: off
        model_outputs = [
            {
                'stride': (30.0, 0.0, 5.0),
                'tokens': np.array([[286, 478, 2633, 760, 420, 2633, 264, 558, 2372, 13, 286, 841, 264, 596, 346, 13, 583, 406, 1547, 281]]),
                'token_timestamps': np.array(
                                  [[23.88, 24.06, 24.06, 24.3, 24.54, 24.72, 24.98, 25.2, 25.36, 25.62,
                                    25.66, 25.8, 26.06, 26.26, 26.34, 26.48, 26.52, 26.72, 26.86, 27.08]])
            },
            {
                'stride': (10.0075, 5.0, 0.0),
                'tokens': np.array([[2633, 6385, 286, 478, 2633, 264, 558, 2372, 286, 841, 264, 4588, 457, 406, 1547, 281, 652, 385, 605, 493]]),
                'token_timestamps': np.array(
                                  [[4.12, 4.32, 4.58, 4.76, 4.84, 4.9, 5.2, 5.36, 5.62, 5.82,
                                    6.02, 6.26, 6.48, 6.74, 6.86, 7.08, 7.32, 7.42, 7.66, 7.8]])
            }
        ]
        # fmt: on

        result = tokenizer._decode_asr(
            model_outputs, return_timestamps="word", return_language=False, time_precision=0.02
        )

        EXPECTED_OUTPUT = (
            " ofectjoy knowocjoy sace threat.ublic s influencept Lians anoryusical",
            {
                "chunks": [
                    {"text": " ofectjoy", "timestamp": (23.88, 24.3)},
                    {"text": " knowocjoy", "timestamp": (24.3, 24.98)},
                    {"text": " sace", "timestamp": (24.98, 25.36)},
                    {"text": " threat", "timestamp": (25.36, 25.62)},
                    {"text": ".ublic", "timestamp": (25.62, 26.02)},
                    {"text": " s", "timestamp": (26.02, 26.26)},
                    {"text": " influencept", "timestamp": (26.26, 26.74)},
                    {"text": " Lians", "timestamp": (26.74, 27.08)},
                    {"text": " anoryusical", "timestamp": (27.08, 27.8)},
                ]
            },
        )

        self.assertEqual(result, EXPECTED_OUTPUT)


class SpeechToTextTokenizerMultilinguialTest(unittest.TestCase):
    checkpoint_name = "openai/whisper-small.en"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(cls.checkpoint_name)
        return cls

    def test_tokenizer_equivalence(self):
        text = "다람쥐 헌 쳇바퀴에 타고파"
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="korean")
        monolingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")

        monolingual_tokens = monolingual_tokenizer.encode(text, add_special_tokens=False)
        multilingual_tokens = multilingual_tokenizer.encode(text, add_special_tokens=False)

        assert monolingual_tokenizer.decode(monolingual_tokens) == text
        assert multilingual_tokenizer.decode(multilingual_tokens) == text
        assert len(monolingual_tokens) > len(multilingual_tokens)

        # fmt: off
        EXPECTED_ENG = [
            46695, 97, 167, 252, 234, 168, 98, 238, 220, 169,
            245, 234, 23821, 111, 229, 167, 108, 242, 169, 222,
            112, 168, 245, 238, 220, 169, 225, 222, 166, 111,
            254, 169, 234, 234
        ]
        EXPECTED_MULTI = [
            9835, 22855, 168, 98, 238, 13431, 234, 43517, 229, 47053,
            169, 222, 19086, 19840, 1313, 17974
        ]
        # fmt: on

        self.assertListEqual(monolingual_tokens, EXPECTED_ENG)
        self.assertListEqual(multilingual_tokens, EXPECTED_MULTI)

    def test_tokenizer_special(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="english", task="transcribe"
        )
        text = "Hey! How are you feeling? J'ai l'impression que 郷さん est prêt"

        multilingual_tokens = multilingual_tokenizer.encode(text)

        # fmt: off
        # format: <|startoftranscript|> <|lang-id|> <|task|> <|notimestamps|> ... transcription ids ... <|endoftext|>
        EXPECTED_MULTI = [
            START_OF_TRANSCRIPT, EN_CODE, TRANSCRIBE, NOTIMESTAMPS, 7057, 0, 1012, 366, 291,
            2633, 30, 508, 6, 1301, 287, 6, 36107, 631, 220, 11178,
            115, 15567, 871, 44393, END_OF_TRANSCRIPT
        ]
        EXPECTED_SPECIAL_TEXT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>Hey! How are you feeling? "
            "J'ai l'impression que 郷さん est prêt<|endoftext|>"
        )
        # fmt: on

        self.assertListEqual(multilingual_tokens, EXPECTED_MULTI)

        special_transcript = multilingual_tokenizer.decode(multilingual_tokens, skip_special_tokens=False)
        self.assertEqual(special_transcript, EXPECTED_SPECIAL_TEXT)

        transcript = multilingual_tokenizer.decode(multilingual_tokens, skip_special_tokens=True)
        self.assertEqual(transcript, text)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 50257)

    # Copied from tests.models.speech_to_text.test_tokenization_speech_to_text.SpeechToTextTokenizerMultilinguialTest.test_tokenizer_decode_ignores_language_codes
    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(ES_CODE, self.tokenizer.all_special_ids)
        generated_ids = [ES_CODE, 4, 1601, 47, 7647, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_spanish = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_spanish)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_batch_encoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="spanish", task="translate"
        )
        batch = ["El gato ", "El gato se sentó"]
        batch_output = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids

        # fmt: off
        EXPECTED_MULTI = [
            [START_OF_TRANSCRIPT, ES_CODE, TRANSLATE, NOTIMESTAMPS, 17356, 290, 2513, 220,
             END_OF_TRANSCRIPT, END_OF_TRANSCRIPT, END_OF_TRANSCRIPT],
            [START_OF_TRANSCRIPT, ES_CODE, TRANSLATE, NOTIMESTAMPS, 17356, 290, 2513, 369,
             2279, 812, END_OF_TRANSCRIPT]
        ]
        # fmt: on

        self.assertListEqual(batch_output, EXPECTED_MULTI)

    def test_set_prefix_tokens(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="spanish", task="translate"
        )

        # change the language prefix token from Spanish to English
        multilingual_tokenizer.set_prefix_tokens(language="english")

        batch = ["the cat", "the cat sat"]
        batch_output = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids

        # fmt: off
        EXPECTED_MULTI = [
            [START_OF_TRANSCRIPT, EN_CODE, TRANSLATE, NOTIMESTAMPS, 3322, 3857,
             END_OF_TRANSCRIPT, END_OF_TRANSCRIPT],
            [START_OF_TRANSCRIPT, EN_CODE, TRANSLATE, NOTIMESTAMPS, 3322, 3857,
             3227, END_OF_TRANSCRIPT]
        ]
        # fmt: on

        self.assertListEqual(batch_output, EXPECTED_MULTI)

    def test_batch_encoding_decoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
        batch = ["hola güey", "que onda"]
        batch_encoding = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids
        transcription = multilingual_tokenizer.batch_decode(batch_encoding, skip_special_tokens=True)
        self.assertListEqual(batch, transcription)

    def test_offset_decoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        # fmt: off
        INPUT_TOKENS = [
            50258, 50259, 50359, 50364, 441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724, 50724, 366,
            382, 4048, 382, 257, 361, 18459, 13065, 13, 2221, 13,
            7145, 74, 325, 38756, 311, 29822, 7563, 412, 472, 709,
            294, 264, 51122, 51122, 912, 636, 300, 2221, 13, 2741,
            5767, 1143, 281, 7319, 702, 7798, 13, 400, 2221, 13,
            2619, 4004, 811, 2709, 702, 51449, 51449, 50257
        ]
        # fmt: on
        output = multilingual_tokenizer.decode(INPUT_TOKENS, output_offsets=True)["offsets"]

        self.assertEqual(
            output,
            [
                {
                    "text": (
                        " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles"
                    ),
                    "timestamp": (0.0, 7.2),
                },
                {
                    "text": (
                        " are as national as a jingo poem. Mr. Birkut Foster's landscapes smile at one much in the"
                    ),
                    "timestamp": (7.2, 15.16),
                },
                {
                    "text": " same way that Mr. Carker used to flash his teeth. And Mr. John Colier gives his",
                    "timestamp": (15.16, 21.7),
                },
            ],
        )
        # test `decode_with_offsets`
        output = multilingual_tokenizer.decode(INPUT_TOKENS, decode_with_timestamps=True)
        self.assertEqual(
            output,
            "<|startoftranscript|><|en|><|transcribe|><|0.00|> Lennils, pictures are a sort of upguards and atom"
            " paintings, and Mason's exquisite idles<|7.20|><|7.20|> are as national as a jingo poem. Mr. Birkut"
            " Foster's landscapes smile at one much in the<|15.16|><|15.16|> same way that Mr. Carker used to flash"
            " his teeth. And Mr. John Colier gives his<|21.70|><|21.70|><|endoftext|>",
        )
        # test a single sequence with timestamps
        # fmt: off
        INPUT_TOKENS = [
            50364, 441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724
        ]
        # fmt: on

        output = multilingual_tokenizer.decode(INPUT_TOKENS, output_offsets=True)["offsets"]
        self.assertEqual(
            output[0],
            {
                "text": " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles",
                "timestamp": (0.0, 7.2),
            },
        )

        # test a sequence without a single timestamps
        # fmt: off
        INPUT_TOKENS = [
            441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724
        ]
        # fmt: on

        output = multilingual_tokenizer.decode(INPUT_TOKENS, output_offsets=True)["offsets"]
        self.assertEqual(output, [])

    def test_convert_to_list_np(self):
        test_list = [[1, 2, 3], [4, 5, 6]]

        # Test with an already converted list
        self.assertListEqual(WhisperTokenizer._convert_to_list(test_list), test_list)
        self.assertListEqual(WhisperTokenizerFast._convert_to_list(test_list), test_list)

        # Test with a numpy array
        np_array = np.array(test_list)
        self.assertListEqual(WhisperTokenizer._convert_to_list(np_array), test_list)
        self.assertListEqual(WhisperTokenizerFast._convert_to_list(np_array), test_list)

    @require_tf
    def test_convert_to_list_tf(self):
        import tensorflow as tf

        test_list = [[1, 2, 3], [4, 5, 6]]
        tf_tensor = tf.constant(test_list)
        self.assertListEqual(WhisperTokenizer._convert_to_list(tf_tensor), test_list)
        self.assertListEqual(WhisperTokenizerFast._convert_to_list(tf_tensor), test_list)

    @require_flax
    def test_convert_to_list_jax(self):
        import jax.numpy as jnp

        test_list = [[1, 2, 3], [4, 5, 6]]
        jax_array = jnp.array(test_list)
        self.assertListEqual(WhisperTokenizer._convert_to_list(jax_array), test_list)
        self.assertListEqual(WhisperTokenizerFast._convert_to_list(jax_array), test_list)

    @require_torch
    def test_convert_to_list_pt(self):
        import torch

        test_list = [[1, 2, 3], [4, 5, 6]]
        torch_tensor = torch.tensor(test_list)
        self.assertListEqual(WhisperTokenizer._convert_to_list(torch_tensor), test_list)
        self.assertListEqual(WhisperTokenizerFast._convert_to_list(torch_tensor), test_list)
