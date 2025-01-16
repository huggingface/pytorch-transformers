# coding=utf-8
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
"""
Please note that Pop2PianoTokenizer is too far from our usual tokenizers and thus cannot use the TokenizerTesterMixin class.
"""

import os
import pickle
import shutil
import tempfile
import unittest

from transformers.feature_extraction_utils import BatchFeature
from transformers.testing_utils import (
    is_pretty_midi_available,
    is_torch_available,
    require_pretty_midi,
    require_torch,
)
from transformers.tokenization_utils import BatchEncoding


if is_torch_available():
    import torch


requirements_available = is_torch_available() and is_pretty_midi_available()
if requirements_available:
    import pretty_midi

    from transformers import Pop2PianoTokenizer


@require_torch
@require_pretty_midi
class Pop2PianoTokenizerTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

    def get_input_notes(self):
        notes = [
            [
                pretty_midi.Note(start=0.441179, end=2.159456, pitch=70, velocity=77),
                pretty_midi.Note(start=0.673379, end=0.905578, pitch=73, velocity=77),
                pretty_midi.Note(start=0.905578, end=2.159456, pitch=73, velocity=77),
                pretty_midi.Note(start=1.114558, end=2.159456, pitch=78, velocity=77),
                pretty_midi.Note(start=1.323537, end=1.532517, pitch=80, velocity=77),
            ],
            [
                pretty_midi.Note(start=0.441179, end=2.159456, pitch=70, velocity=77),
            ],
        ]

        return notes

    def test_call(self):
        notes = self.get_input_notes()

        output = self.tokenizer(
            notes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=10,
            return_attention_mask=True,
        )

        # check the output type
        self.assertTrue(isinstance(output, BatchEncoding))

        # check the values
        expected_output_token_ids = torch.tensor(
            [[134, 133, 74, 135, 77, 132, 77, 133, 77, 82], [134, 133, 74, 136, 132, 74, 134, 134, 134, 134]]
        )
        expected_output_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

        torch.testing.assert_close(output["token_ids"], expected_output_token_ids, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output["attention_mask"], expected_output_attention_mask, rtol=1e-4, atol=1e-4)

    def test_batch_decode(self):
        # test batch decode with model, feature-extractor outputs(beatsteps, extrapolated_beatstep)

        # Please note that this test does not test the accuracy of the outputs, instead it is designed to make sure that
        # the tokenizer's batch_decode can deal with attention_mask in feature-extractor outputs. For the accuracy check
        # please see the `test_batch_decode_outputs` test.

        model_output = torch.concatenate(
            [
                torch.randint(size=[120, 96], low=0, high=70, dtype=torch.long),
                torch.zeros(size=[1, 96], dtype=torch.long),
                torch.randint(size=[50, 96], low=0, high=40, dtype=torch.long),
                torch.zeros(size=[1, 96], dtype=torch.long),
            ],
            axis=0,
        )
        input_features = BatchFeature(
            {
                "beatsteps": torch.ones([2, 955]),
                "extrapolated_beatstep": torch.ones([2, 1000]),
                "attention_mask": torch.concatenate(
                    [
                        torch.ones([120, 96], dtype=torch.long),
                        torch.zeros([1, 96], dtype=torch.long),
                        torch.ones([50, 96], dtype=torch.long),
                        torch.zeros([1, 96], dtype=torch.long),
                    ],
                    axis=0,
                ),
                "attention_mask_beatsteps": torch.ones([2, 955]),
                "attention_mask_extrapolated_beatstep": torch.ones([2, 1000]),
            }
        )

        output = self.tokenizer.batch_decode(token_ids=model_output, feature_extractor_output=input_features)[
            "pretty_midi_objects"
        ]

        # check length
        self.assertTrue(len(output) == 2)

        # check object type
        self.assertTrue(isinstance(output[0], pretty_midi.pretty_midi.PrettyMIDI))
        self.assertTrue(isinstance(output[1], pretty_midi.pretty_midi.PrettyMIDI))

    def test_batch_decode_outputs(self):
        # test batch decode with model, feature-extractor outputs(beatsteps, extrapolated_beatstep)

        # Please note that this test tests the accuracy of the outputs of the tokenizer's `batch_decode` method.

        model_output = torch.tensor(
            [
                [134, 133, 74, 135, 77, 82, 84, 136, 132, 74, 77, 82, 84],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        input_features = BatchEncoding(
            {
                "beatsteps": torch.tensor([[0.0697, 0.1103, 0.1509, 0.1916]]),
                "extrapolated_beatstep": torch.tensor([[0.0000, 0.0406, 0.0813, 0.1219]]),
            }
        )

        output = self.tokenizer.batch_decode(token_ids=model_output, feature_extractor_output=input_features)

        # check outputs
        self.assertEqual(len(output["notes"]), 4)

        predicted_start_timings, predicted_end_timings = [], []
        for i in output["notes"]:
            predicted_start_timings.append(i.start)
            predicted_end_timings.append(i.end)

        # Checking note start timings
        expected_start_timings = torch.tensor(
            [
                0.069700,
                0.110300,
                0.110300,
                0.110300,
            ]
        )
        predicted_start_timings = torch.tensor(predicted_start_timings)

        torch.testing.assert_close(expected_start_timings, predicted_start_timings, rtol=1e-4, atol=1e-4)

        # Checking note end timings
        expected_end_timings = torch.tensor(
            [
                0.191600,
                0.191600,
                0.191600,
                0.191600,
            ]
        )
        predicted_end_timings = torch.tensor(predicted_end_timings)

        torch.testing.assert_close(expected_end_timings, predicted_end_timings, rtol=1e-4, atol=1e-4)

    def test_get_vocab(self):
        vocab_dict = self.tokenizer.get_vocab()
        self.assertIsInstance(vocab_dict, dict)
        self.assertGreaterEqual(len(self.tokenizer), len(vocab_dict))

        vocab = [self.tokenizer.convert_ids_to_tokens(i) for i in range(len(self.tokenizer))]
        self.assertEqual(len(vocab), len(self.tokenizer))

        self.tokenizer.add_tokens(["asdfasdfasdfasdf"])
        vocab = [self.tokenizer.convert_ids_to_tokens(i) for i in range(len(self.tokenizer))]
        self.assertEqual(len(vocab), len(self.tokenizer))

    def test_save_and_load_tokenizer(self):
        tmpdirname = tempfile.mkdtemp()

        sample_notes = self.get_input_notes()

        self.tokenizer.add_tokens(["bim", "bambam"])
        additional_special_tokens = self.tokenizer.additional_special_tokens
        additional_special_tokens.append("new_additional_special_token")
        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        before_token_ids = self.tokenizer(sample_notes)["token_ids"]
        before_vocab = self.tokenizer.get_vocab()
        self.tokenizer.save_pretrained(tmpdirname)

        after_tokenizer = self.tokenizer.__class__.from_pretrained(tmpdirname)
        after_token_ids = after_tokenizer(sample_notes)["token_ids"]
        after_vocab = after_tokenizer.get_vocab()
        self.assertDictEqual(before_vocab, after_vocab)
        self.assertListEqual(before_token_ids, after_token_ids)
        self.assertIn("bim", after_vocab)
        self.assertIn("bambam", after_vocab)
        self.assertIn("new_additional_special_token", after_tokenizer.additional_special_tokens)

        shutil.rmtree(tmpdirname)

    def test_pickle_tokenizer(self):
        tmpdirname = tempfile.mkdtemp()

        notes = self.get_input_notes()
        subwords = self.tokenizer(notes)["token_ids"]

        filename = os.path.join(tmpdirname, "tokenizer.bin")
        with open(filename, "wb") as handle:
            pickle.dump(self.tokenizer, handle)

        with open(filename, "rb") as handle:
            tokenizer_new = pickle.load(handle)

        subwords_loaded = tokenizer_new(notes)["token_ids"]

        self.assertListEqual(subwords, subwords_loaded)

    def test_padding_side_in_kwargs(self):
        tokenizer_p = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano", padding_side="left")
        self.assertEqual(tokenizer_p.padding_side, "left")

        tokenizer_p = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano", padding_side="right")
        self.assertEqual(tokenizer_p.padding_side, "right")

        self.assertRaises(
            ValueError,
            Pop2PianoTokenizer.from_pretrained,
            "sweetcocoa/pop2piano",
            padding_side="unauthorized",
        )

    def test_truncation_side_in_kwargs(self):
        tokenizer_p = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano", truncation_side="left")
        self.assertEqual(tokenizer_p.truncation_side, "left")

        tokenizer_p = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano", truncation_side="right")
        self.assertEqual(tokenizer_p.truncation_side, "right")

        self.assertRaises(
            ValueError,
            Pop2PianoTokenizer.from_pretrained,
            "sweetcocoa/pop2piano",
            truncation_side="unauthorized",
        )

    def test_right_and_left_padding(self):
        tokenizer = self.tokenizer
        notes = self.get_input_notes()
        notes = notes[0]
        max_length = 20

        padding_idx = tokenizer.pad_token_id

        # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
        tokenizer.padding_side = "right"
        padded_notes = tokenizer(notes, padding="max_length", max_length=max_length)["token_ids"]
        padded_notes_length = len(padded_notes)
        notes_without_padding = tokenizer(notes, padding="do_not_pad")["token_ids"]
        padding_size = max_length - len(notes_without_padding)

        self.assertEqual(padded_notes_length, max_length)
        self.assertEqual(notes_without_padding + [padding_idx] * padding_size, padded_notes)

        # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
        tokenizer.padding_side = "left"
        padded_notes = tokenizer(notes, padding="max_length", max_length=max_length)["token_ids"]
        padded_notes_length = len(padded_notes)
        notes_without_padding = tokenizer(notes, padding="do_not_pad")["token_ids"]
        padding_size = max_length - len(notes_without_padding)

        self.assertEqual(padded_notes_length, max_length)
        self.assertEqual([padding_idx] * padding_size + notes_without_padding, padded_notes)

        # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
        notes_without_padding = tokenizer(notes)["token_ids"]

        tokenizer.padding_side = "right"
        padded_notes_right = tokenizer(notes, padding=False)["token_ids"]
        self.assertEqual(len(padded_notes_right), len(notes_without_padding))
        self.assertEqual(padded_notes_right, notes_without_padding)

        tokenizer.padding_side = "left"
        padded_notes_left = tokenizer(notes, padding="longest")["token_ids"]
        self.assertEqual(len(padded_notes_left), len(notes_without_padding))
        self.assertEqual(padded_notes_left, notes_without_padding)

        tokenizer.padding_side = "right"
        padded_notes_right = tokenizer(notes, padding="longest")["token_ids"]
        self.assertEqual(len(padded_notes_right), len(notes_without_padding))
        self.assertEqual(padded_notes_right, notes_without_padding)

        tokenizer.padding_side = "left"
        padded_notes_left = tokenizer(notes, padding=False)["token_ids"]
        self.assertEqual(len(padded_notes_left), len(notes_without_padding))
        self.assertEqual(padded_notes_left, notes_without_padding)

    def test_right_and_left_truncation(self):
        tokenizer = self.tokenizer
        notes = self.get_input_notes()
        notes = notes[0]
        truncation_size = 3

        # RIGHT TRUNCATION - Check that it correctly truncates when a maximum length is specified along with the truncation flag set to True
        tokenizer.truncation_side = "right"
        full_encoded_notes = tokenizer(notes)["token_ids"]
        full_encoded_notes_length = len(full_encoded_notes)
        truncated_notes = tokenizer(notes, max_length=full_encoded_notes_length - truncation_size, truncation=True)[
            "token_ids"
        ]
        self.assertEqual(full_encoded_notes_length, len(truncated_notes) + truncation_size)
        self.assertEqual(full_encoded_notes[:-truncation_size], truncated_notes)

        # LEFT TRUNCATION - Check that it correctly truncates when a maximum length is specified along with the truncation flag set to True
        tokenizer.truncation_side = "left"
        full_encoded_notes = tokenizer(notes)["token_ids"]
        full_encoded_notes_length = len(full_encoded_notes)
        truncated_notes = tokenizer(notes, max_length=full_encoded_notes_length - truncation_size, truncation=True)[
            "token_ids"
        ]
        self.assertEqual(full_encoded_notes_length, len(truncated_notes) + truncation_size)
        self.assertEqual(full_encoded_notes[truncation_size:], truncated_notes)

        # RIGHT & LEFT TRUNCATION - Check that nothing is done for 'longest' and 'no_truncation'
        tokenizer.truncation_side = "right"
        truncated_notes_right = tokenizer(notes, truncation=True)["token_ids"]
        self.assertEqual(full_encoded_notes_length, len(truncated_notes_right))
        self.assertEqual(full_encoded_notes, truncated_notes_right)

        tokenizer.truncation_side = "left"
        truncated_notes_left = tokenizer(notes, truncation="longest_first")["token_ids"]
        self.assertEqual(len(truncated_notes_left), full_encoded_notes_length)
        self.assertEqual(truncated_notes_left, full_encoded_notes)

        tokenizer.truncation_side = "right"
        truncated_notes_right = tokenizer(notes, truncation="longest_first")["token_ids"]
        self.assertEqual(len(truncated_notes_right), full_encoded_notes_length)
        self.assertEqual(truncated_notes_right, full_encoded_notes)

        tokenizer.truncation_side = "left"
        truncated_notes_left = tokenizer(notes, truncation=True)["token_ids"]
        self.assertEqual(len(truncated_notes_left), full_encoded_notes_length)
        self.assertEqual(truncated_notes_left, full_encoded_notes)

    def test_padding_to_multiple_of(self):
        notes = self.get_input_notes()

        if self.tokenizer.pad_token is None:
            self.skipTest(reason="No padding token.")
        else:
            normal_tokens = self.tokenizer(notes[0], padding=True, pad_to_multiple_of=8)
            for key, value in normal_tokens.items():
                self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

            normal_tokens = self.tokenizer(notes[0], pad_to_multiple_of=8)
            for key, value in normal_tokens.items():
                self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

            # Should also work with truncation
            normal_tokens = self.tokenizer(notes[0], padding=True, truncation=True, pad_to_multiple_of=8)
            for key, value in normal_tokens.items():
                self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

            # truncation to something which is not a multiple of pad_to_multiple_of raises an error
            self.assertRaises(
                ValueError,
                self.tokenizer.__call__,
                notes[0],
                padding=True,
                truncation=True,
                max_length=12,
                pad_to_multiple_of=8,
            )

    def test_padding_with_attention_mask(self):
        if self.tokenizer.pad_token is None:
            self.skipTest(reason="No padding token.")
        if "attention_mask" not in self.tokenizer.model_input_names:
            self.skipTest(reason="This model does not use attention mask.")

        features = [
            {"token_ids": [1, 2, 3, 4, 5, 6], "attention_mask": [1, 1, 1, 1, 1, 0]},
            {"token_ids": [1, 2, 3], "attention_mask": [1, 1, 0]},
        ]
        padded_features = self.tokenizer.pad(features)
        if self.tokenizer.padding_side == "right":
            self.assertListEqual(padded_features["attention_mask"], [[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]])
        else:
            self.assertListEqual(padded_features["attention_mask"], [[1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0]])
