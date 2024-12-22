# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
import inspect
import os
import shutil
import tempfile
import unittest
from typing import List

import numpy as np
import pandas as pd
from parameterized import parameterized

from transformers import AddedToken
from transformers.models.tapas.tokenization_tapas import (
    VOCAB_FILES_NAMES,
    BasicTokenizer,
    TapasTokenizer,
    WordpieceTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)
from transformers.testing_utils import (
    is_pt_tf_cross_test,
    require_pandas,
    require_tensorflow_probability,
    require_tokenizers,
    require_torch,
    slow,
)

from ...test_tokenization_common import TokenizerTesterMixin, filter_non_english, merge_model_tokenizer_mappings


@require_tokenizers
@require_pandas
class TapasTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/tapas-large-finetuned-sqa"
    tokenizer_class = TapasTokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = False

    def get_table(
        self,
        tokenizer: TapasTokenizer,
        length=5,
    ):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]

        if length == 0:
            data = {}
        else:
            data = {toks[0]: [toks[tok] for tok in range(1, length)]}

        table = pd.DataFrame.from_dict(data)

        return table

    def get_table_and_query(
        self,
        tokenizer: TapasTokenizer,
        length=5,
    ):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]
        table = self.get_table(tokenizer, length=length - 3)
        query = " ".join(toks[:3])

        return table, query

    def get_clean_sequence(
        self,
        tokenizer: TapasTokenizer,
        with_prefix_space=False,
        max_length=20,
        min_length=5,
        empty_table: bool = False,
        add_special_tokens: bool = True,
        return_table_and_query: bool = False,
    ):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]

        if empty_table:
            table = pd.DataFrame.from_dict({})
            query = " ".join(toks[:min_length])
        else:
            data = {toks[0]: [toks[tok] for tok in range(1, min_length - 3)]}
            table = pd.DataFrame.from_dict(data)
            query = " ".join(toks[:3])

        output_ids = tokenizer.encode(table, query, add_special_tokens=add_special_tokens)
        output_txt = tokenizer.decode(output_ids)

        assert len(output_ids) >= min_length, "Update the code to generate the sequences so that they are larger"
        assert len(output_ids) <= max_length, "Update the code to generate the sequences so that they are smaller"

        if return_table_and_query:
            return output_txt, output_ids, table, query

        return output_txt, output_ids

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00e9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    @require_tensorflow_probability
    @slow
    def test_tf_encode_plus_sent_to_model(self):
        from transformers import TF_MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(TF_MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    self.skipTest(f"{tokenizer.__class__} is not in the MODEL_TOKENIZER_MAPPING")

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    self.skipTest(reason="Model is an encoder-decoder or does not have a pad token id set")

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                self.assertGreaterEqual(model.config.vocab_size, len(tokenizer))

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                table = self.get_table(tokenizer, length=0)
                encoded_sequence = tokenizer.encode_plus(table, sequence, return_tensors="tf")
                batch_encoded_sequence = tokenizer.batch_encode_plus(table, [sequence, sequence], return_tensors="tf")

                # This should not fail
                model(encoded_sequence)
                model(batch_encoded_sequence)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "UNwant\u00e9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # With lower casing
        tokenizer = self.get_tokenizer(do_lower_case=True)
        rust_tokenizer = self.get_rust_tokenizer(do_lower_case=True)

        sequence = "UNwant\u00e9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    @unittest.skip(reason="Chat template tests don't play well with table/layout models.")
    def test_chat_template_batched(self):
        pass

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535a\u63a8zz"), ["ah", "\u535a", "\u63a8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["hello", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hällo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["h\u00e9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00e9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HäLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HaLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"), ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"]
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]

        vocab = {}
        for i, token in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("unwanted running"), ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_is_whitespace(self):
        self.assertTrue(_is_whitespace(" "))
        self.assertTrue(_is_whitespace("\t"))
        self.assertTrue(_is_whitespace("\r"))
        self.assertTrue(_is_whitespace("\n"))
        self.assertTrue(_is_whitespace("\u00a0"))

        self.assertFalse(_is_whitespace("A"))
        self.assertFalse(_is_whitespace("-"))

    def test_is_control(self):
        self.assertTrue(_is_control("\u0005"))

        self.assertFalse(_is_control("A"))
        self.assertFalse(_is_control(" "))
        self.assertFalse(_is_control("\t"))
        self.assertFalse(_is_control("\r"))

    def test_is_punctuation(self):
        self.assertTrue(_is_punctuation("-"))
        self.assertTrue(_is_punctuation("$"))
        self.assertTrue(_is_punctuation("`"))
        self.assertTrue(_is_punctuation("."))

        self.assertFalse(_is_punctuation("A"))
        self.assertFalse(_is_punctuation(" "))

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], ["[EMPTY]"], ["[UNK]"]]
        )

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("google/tapas-base-finetuned-wtq")

        empty_table = self.get_table(tokenizer, length=0)
        table = self.get_table(tokenizer, length=10)

        text = tokenizer.encode(table, add_special_tokens=False)
        text_2 = tokenizer.encode(empty_table, "multi-sequence build", add_special_tokens=False)

        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_pair == [101] + text + [102] + text_2

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                sentence = f"A, naïve {tokenizer_r.mask_token} AllenNLP sentence."
                tokens = tokenizer_r.encode_plus(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                do_lower_case = tokenizer_r.do_lower_case if hasattr(tokenizer_r, "do_lower_case") else False
                expected_results = (
                    [
                        ((0, 0), tokenizer_r.cls_token),
                        ((0, 1), "A"),
                        ((1, 2), ","),
                        ((3, 5), "na"),
                        ((5, 6), "##ï"),
                        ((6, 8), "##ve"),
                        ((9, 15), tokenizer_r.mask_token),
                        ((16, 21), "Allen"),
                        ((21, 23), "##NL"),
                        ((23, 24), "##P"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer_r.sep_token),
                    ]
                    if not do_lower_case
                    else [
                        ((0, 0), tokenizer_r.cls_token),
                        ((0, 1), "a"),
                        ((1, 2), ","),
                        ((3, 8), "naive"),
                        ((9, 15), tokenizer_r.mask_token),
                        ((16, 21), "allen"),
                        ((21, 23), "##nl"),
                        ((23, 24), "##p"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer_r.sep_token),
                    ]
                )

                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer_r.convert_ids_to_tokens(tokens["input_ids"])
                )
                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

    def test_add_special_tokens(self):
        tokenizers: List[TapasTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_table = self.get_table(tokenizer, length=0)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(input_table, special_token, add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                decoded = tokenizer.decode(encoded_special_token, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_add_tokens_tokenizer(self):
        tokenizers: List[TapasTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode(table, "aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    table,
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    add_special_tokens=False,
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    @require_tokenizers
    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)

                new_toks = [AddedToken("[ABC]", normalized=False), AddedToken("[DEF]", normalized=False)]
                tokenizer.add_tokens(new_toks)
                input = "[ABC][DEF][ABC][DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] [DEF]"
                else:
                    output = input
                encoded = tokenizer.encode(table, input, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    @parameterized.expand([(True,), (False,)])
    def test_encode_plus_with_padding(self, use_padding_as_call_kwarg: bool):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequence = "Sequence"

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id
                token_type_padding_idx = tokenizer.pad_token_type_id

                encoded_sequence = tokenizer.encode_plus(table, sequence, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test 'longest' and 'no_padding' don't do anything
                tokenizer.padding_side = "right"

                not_padded_sequence = tokenizer.encode_plus(
                    table,
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                assert sequence_length == not_padded_sequence_length
                assert input_ids == not_padded_input_ids
                assert special_tokens_mask == not_padded_special_tokens_mask

                not_padded_sequence = tokenizer.encode_plus(
                    table,
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                assert sequence_length == not_padded_sequence_length
                assert input_ids == not_padded_input_ids
                assert special_tokens_mask == not_padded_special_tokens_mask

                # Test right padding
                tokenizer_kwargs_right = {
                    "max_length": sequence_length + padding_size,
                    "padding": "max_length",
                    "return_special_tokens_mask": True,
                }

                if not use_padding_as_call_kwarg:
                    tokenizer.padding_side = "right"
                else:
                    tokenizer_kwargs_right["padding_side"] = "right"

                right_padded_sequence = tokenizer.encode_plus(table, sequence, **tokenizer_kwargs_right)
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                assert sequence_length + padding_size == right_padded_sequence_length
                assert input_ids + [padding_idx] * padding_size == right_padded_input_ids
                assert special_tokens_mask + [1] * padding_size == right_padded_special_tokens_mask

                # Test left padding
                tokenizer_kwargs_left = {
                    "max_length": sequence_length + padding_size,
                    "padding": "max_length",
                    "return_special_tokens_mask": True,
                }

                if not use_padding_as_call_kwarg:
                    tokenizer.padding_side = "left"
                else:
                    tokenizer_kwargs_left["padding_side"] = "left"

                left_padded_sequence = tokenizer.encode_plus(table, sequence, **tokenizer_kwargs_left)
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                assert sequence_length + padding_size == left_padded_sequence_length
                assert [padding_idx] * padding_size + input_ids == left_padded_input_ids
                assert [1] * padding_size + special_tokens_mask == left_padded_special_tokens_mask

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

                    assert (
                        token_type_ids + [[token_type_padding_idx] * 7] * padding_size == right_padded_token_type_ids
                    )
                    assert [[token_type_padding_idx] * 7] * padding_size + token_type_ids == left_padded_token_type_ids

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    assert attention_mask + [0] * padding_size == right_padded_attention_mask
                    assert [0] * padding_size + attention_mask == left_padded_attention_mask

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(table, input_text, add_special_tokens=False)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2, output_text)

    def test_mask_output(self):
        tokenizers = self.get_tokenizers(fast=False, do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table, query = self.get_table_and_query(tokenizer)

                if (
                    tokenizer.build_inputs_with_special_tokens.__qualname__.split(".")[0] != "PreTrainedTokenizer"
                    and "token_type_ids" in tokenizer.model_input_names
                ):
                    information = tokenizer.encode_plus(table, query, add_special_tokens=True)
                    sequences, mask = information["input_ids"], information["token_type_ids"]
                    self.assertEqual(len(sequences), len(mask))

    @unittest.skip(reason="TAPAS tokenizer only handles two sequences.")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip(reason="TAPAS tokenizer only handles two sequences.")
    def test_maximum_encoding_length_single_input(self):
        pass

    def test_number_of_added_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table, query = self.get_table_and_query(tokenizer)

                sequences = tokenizer.encode(table, query, add_special_tokens=False)
                attached_sequences = tokenizer.encode(table, query, add_special_tokens=True)

                # Method is implemented (e.g. not GPT-2)
                if len(attached_sequences) != 2:
                    self.assertEqual(
                        tokenizer.num_special_tokens_to_add(pair=True), len(attached_sequences) - len(sequences)
                    )

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be removed when `pad_to_max_length` will be deprecated"""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer)
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence = tokenizer.encode(
                    table, sequence, max_length=sequence_length + padding_size, padding=True
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(table, sequence, pad_to_max_length=True)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                # Test not batched
                table = self.get_table(tokenizer, length=0)
                encoded_sequences_1 = tokenizer.encode_plus(table, sequences[0])
                encoded_sequences_2 = tokenizer(table, sequences[0])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                table = self.get_table(tokenizer, length=10)
                encoded_sequences_1 = tokenizer.encode_plus(table, sequences[1])
                encoded_sequences_2 = tokenizer(table, sequences[1])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                table = self.get_table(tokenizer, length=0)
                encoded_sequences_1 = tokenizer.batch_encode_plus(table, sequences)
                encoded_sequences_2 = tokenizer(table, sequences)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                encoded_sequences = [tokenizer.encode_plus(table, sequence) for sequence in sequences]
                encoded_sequences_batch = tokenizer.batch_encode_plus(table, sequences, padding=False)
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

                maximum_length = len(
                    max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len)
                )

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences_padded = [
                    tokenizer.encode_plus(table, sequence, max_length=maximum_length, padding="max_length")
                    for sequence in sequences
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode_plus(table, sequences, padding=True)
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(table, sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    table, sequences, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(table, sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    table, sequences, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

    @unittest.skip(reason="batch_encode_plus does not handle overflowing tokens.")
    def test_batch_encode_plus_overflowing_tokens(self):
        pass

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        # Right padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode_plus(table, sequence, max_length=max_length, padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    table, sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

        # Left padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.padding_side = "left"
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode_plus(table, sequence, max_length=max_length, padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    table, sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                if tokenizer.pad_token is None:
                    self.skipTest(reason="No padding token.")
                else:
                    empty_tokens = tokenizer(table, padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer(table, "This is a sample input", padding=True, pad_to_multiple_of=8)
                    for key, value in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer(table, "This", pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer(table, "This", padding=True, truncation=True, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

    @unittest.skip(
        reason="TAPAS cannot handle `prepare_for_model` without passing by `encode_plus` or `batch_encode_plus`"
    )
    def test_prepare_for_model(self):
        pass

    def test_tokenizer_slow_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                empty_table = self.get_table(tokenizer, length=0)
                table = self.get_table(tokenizer, length=10)
                encoded_sequence = tokenizer.encode(empty_table, sequence_0, add_special_tokens=False)
                encoded_sequence += tokenizer.encode(table, "", add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    table,
                    sequence_0,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    # add_prefix_space=False,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequence_0 = "Encode this."
                # Testing single inputs
                encoded_sequence = tokenizer.encode(table, sequence_0, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    table, sequence_0, add_special_tokens=True, return_special_tokens_mask=True
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [x for i, x in enumerate(encoded_sequence_w_special) if not special_tokens_mask[i]]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                table = self.get_table(tokenizer, length=0)
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00e9d,running"
                before_tokens = tokenizer.encode(table, sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(table, sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

    @unittest.skip(reason="Not implemented")
    def test_right_and_left_truncation(self):
        pass

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    table, sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    table, sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert [padding_idx] * padding_size + encoded_sequence == padded_sequence

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(table, sequence, padding=True)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(table, sequence, padding="longest")
                padded_sequence_left_length = len(padded_sequence_left)
                assert sequence_length == padded_sequence_left_length
                assert encoded_sequence == padded_sequence_left

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(table, sequence)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(table, sequence, padding=False)
                padded_sequence_left_length = len(padded_sequence_left)
                assert sequence_length == padded_sequence_left_length
                assert encoded_sequence == padded_sequence_left

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                empty_table = self.get_table(tokenizer, length=0)
                seq_0 = "Test this method."

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(empty_table, seq_0, return_token_type_ids=True)

                # Assert that the token type IDs have the same length as the input IDs
                self.assertEqual(len(output["token_type_ids"]), len(output["input_ids"]))

                # Assert that each token type ID has 7 values
                self.assertTrue(all(len(token_type_ids) == 7 for token_type_ids in output["token_type_ids"]))

                # Do the same test as modeling common.
                self.assertIn(0, output["token_type_ids"][0])

    @require_torch
    @slow
    def test_torch_encode_plus_sent_to_model(self):
        import torch

        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    self.skipTest(f"{tokenizer.__class__} is not in the MODEL_TOKENIZER_MAPPING")

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    self.skipTest(reason="Model is an encoder-decoder or has no padding token set.")

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                is_using_common_embeddings = hasattr(model.get_input_embeddings(), "weight")
                assert (
                    (model.get_input_embeddings().weight.shape[0] >= len(tokenizer))
                    if is_using_common_embeddings
                    else True
                )

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                table = self.get_table(tokenizer, length=0)
                encoded_sequence = tokenizer.encode_plus(table, sequence, return_tensors="pt")
                batch_encoded_sequence = tokenizer.batch_encode_plus(table, [sequence, sequence], return_tensors="pt")
                # This should not fail

                with torch.no_grad():  # saves some time
                    model(**encoded_sequence)
                    model(**batch_encoded_sequence)

    @unittest.skip(reason="TAPAS doesn't handle pre-tokenized inputs.")
    def test_pretokenized_inputs(self):
        pass

    @slow
    def test_tapas_truncation_integration_test(self):
        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Age": ["56", "45", "59"],
            "Number of movies": ["87", "53", "69"],
            "Date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
        }
        queries = [
            "When was Brad Pitt born?",
            "Which actor appeared in the least number of movies?",
            "What is the average number of movies?",
        ]
        table = pd.DataFrame.from_dict(data)

        tokenizer = TapasTokenizer.from_pretrained("lysandre/tapas-temporary-repo", model_max_length=512)

        for i in range(12):
            # The table cannot even encode the headers, so raise an error
            with self.assertRaises(ValueError):
                tokenizer.encode(table=table, query=queries[0], max_length=i, truncation="drop_rows_to_fit")

        for i in range(12, 512):
            new_encoded_inputs = tokenizer.encode(
                table=table, query=queries[0], max_length=i, truncation="drop_rows_to_fit"
            )

            # Ensure that the input IDs are less than the max length defined.
            self.assertLessEqual(len(new_encoded_inputs), i)

        tokenizer.model_max_length = 20
        new_encoded_inputs = tokenizer.encode(table=table, query=queries[0], truncation=True)
        dropped_encoded_inputs = tokenizer.encode(table=table, query=queries[0], truncation="drop_rows_to_fit")

        # Ensure that the input IDs are still truncated when no max_length is specified
        self.assertListEqual(new_encoded_inputs, dropped_encoded_inputs)
        self.assertLessEqual(len(new_encoded_inputs), 20)

    @slow
    def test_min_max_question_length(self):
        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Age": ["56", "45", "59"],
            "Number of movies": ["87", "53", "69"],
            "Date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
        }
        queries = "When was Brad Pitt born?"
        table = pd.DataFrame.from_dict(data)

        # test max_question_length
        tokenizer = TapasTokenizer.from_pretrained("lysandre/tapas-temporary-repo", max_question_length=2)

        encoding = tokenizer(table=table, queries=queries)

        # query should not be tokenized as it's longer than the specified max_question_length
        expected_results = [101, 102]

        self.assertListEqual(encoding.input_ids[:2], expected_results)

        # test min_question_length
        tokenizer = TapasTokenizer.from_pretrained("lysandre/tapas-temporary-repo", min_question_length=30)

        encoding = tokenizer(table=table, queries=queries)

        # query should not be tokenized as it's shorter than the specified min_question_length
        expected_results = [101, 102]

        self.assertListEqual(encoding.input_ids[:2], expected_results)

    @is_pt_tf_cross_test
    def test_batch_encode_plus_tensors(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                table = self.get_table(tokenizer, length=0)

                # A Tensor cannot be build by sequences which are not the same size
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, table, sequences, return_tensors="pt")
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, table, sequences, return_tensors="tf")

                if tokenizer.pad_token_id is None:
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        table,
                        sequences,
                        padding=True,
                        return_tensors="pt",
                    )
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        table,
                        sequences,
                        padding="longest",
                        return_tensors="tf",
                    )
                else:
                    pytorch_tensor = tokenizer.batch_encode_plus(table, sequences, padding=True, return_tensors="pt")
                    tensorflow_tensor = tokenizer.batch_encode_plus(
                        table, sequences, padding="longest", return_tensors="tf"
                    )
                    encoded_sequences = tokenizer.batch_encode_plus(table, sequences, padding=True)

                    for key in encoded_sequences.keys():
                        pytorch_value = pytorch_tensor[key].tolist()
                        tensorflow_value = tensorflow_tensor[key].numpy().tolist()
                        encoded_value = encoded_sequences[key]

                        self.assertEqual(pytorch_value, tensorflow_value, encoded_value)

    @slow
    def test_tapas_integration_test(self):
        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Age": ["56", "45", "59"],
            "Number of movies": ["87", "53", "69"],
            "Date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
        }
        queries = [
            "When was Brad Pitt born?",
            "Which actor appeared in the least number of movies?",
            "What is the average number of movies?",
        ]
        table = pd.DataFrame.from_dict(data)

        tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq", model_max_length=512)

        expected_results = {'input_ids':[101,2043,2001,8226,15091,2141,1029,102,5889,2287,2193,1997,5691,3058,1997,4182,8226,15091,5179,6584,2324,2285,3699,14720,4487,6178,9488,3429,5187,2340,2281,3326,2577,18856,7828,3240,5354,6353,1020,2089,3777],'attention_mask':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],'token_type_ids':[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[1,1,0,0,0,0,0],[1,2,0,0,0,0,0],[1,3,0,0,0,0,0],[1,3,0,0,0,0,0],[1,3,0,0,0,0,0],[1,4,0,0,0,0,0],[1,4,0,0,0,0,0],[1,4,0,0,0,0,0],[1,1,1,0,0,0,0],[1,1,1,0,0,0,0],[1,2,1,0,2,2,0],[1,3,1,0,3,1,0],[1,4,1,0,2,2,0],[1,4,1,0,2,2,0],[1,4,1,0,2,2,0],[1,1,2,0,0,0,0],[1,1,2,0,0,0,0],[1,1,2,0,0,0,0],[1,1,2,0,0,0,0],[1,2,2,0,1,3,0],[1,3,2,0,1,3,0],[1,4,2,0,3,1,0],[1,4,2,0,3,1,0],[1,4,2,0,3,1,0],[1,1,3,0,0,0,0],[1,1,3,0,0,0,0],[1,1,3,0,0,0,0],[1,1,3,0,0,0,0],[1,2,3,0,3,1,0],[1,3,3,0,2,2,0],[1,4,3,0,1,3,0],[1,4,3,0,1,3,0],[1,4,3,0,1,3,0]]}  # fmt: skip

        new_encoded_inputs = tokenizer.encode_plus(table=table, query=queries[0])

        self.assertDictEqual(dict(new_encoded_inputs), expected_results)

    @slow
    def test_full_tokenizer(self):
        data = [
            ["Pos", "No", "Driver", "Team", "Laps", "Time/Retired", "Grid", "Points"],
            ["1", "32", "Patrick Carpentier", "Team Player's", "87", "1:48:11.023", "1", "22"],
            ["2", "1", "Bruno Junqueira", "Newman/Haas Racing", "87", "+0.8 secs", "2", "17"],
            ["3", "3", "Paul Tracy", "Team Player's", "87", "+28.6 secs", "3", "14"],
            ["4", "9", "Michel Jourdain, Jr.", "Team Rahal", "87", "+40.8 secs", "13", "12"],
            ["5", "34", "Mario Haberfeld", "Mi-Jack Conquest Racing", "87", "+42.1 secs", "6", "10"],
            ["6", "20", "Oriol Servia", "Patrick Racing", "87", "+1:00.2", "10", "8"],
            ["7", "51", "Adrian Fernandez", "Fernandez Racing", "87", "+1:01.4", "5", "6"],
            ["8", "12", "Jimmy Vasser", "American Spirit Team Johansson", "87", "+1:01.8", "8", "5"],
            ["9", "7", "Tiago Monteiro", "Fittipaldi-Dingman Racing", "86", "+ 1 Lap", "15", "4"],
            ["10", "55", "Mario Dominguez", "Herdez Competition", "86", "+ 1 Lap", "11", "3"],
            ["11", "27", "Bryan Herta", "PK Racing", "86", "+ 1 Lap", "12", "2"],
            ["12", "31", "Ryan Hunter-Reay", "American Spirit Team Johansson", "86", "+ 1 Lap", "17", "1"],
            ["13", "19", "Joel Camathias", "Dale Coyne Racing", "85", "+ 2 Laps", "18", "0"],
            ["14", "33", "Alex Tagliani", "Rocketsports Racing", "85", "+ 2 Laps", "14", "0"],
            ["15", "4", "Roberto Moreno", "Herdez Competition", "85", "+ 2 Laps", "9", "0"],
            ["16", "11", "Geoff Boss", "Dale Coyne Racing", "83", "Mechanical", "19", "0"],
            ["17", "2", "Sebastien Bourdais", "Newman/Haas Racing", "77", "Mechanical", "4", "0"],
            ["18", "15", "Darren Manning", "Walker Racing", "12", "Mechanical", "7", "0"],
            ["19", "5", "Rodolfo Lavin", "Walker Racing", "10", "Mechanical", "16", "0"],
        ]
        query = "what were the drivers names?"
        table = pd.DataFrame.from_records(data[1:], columns=data[0])

        tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq", model_max_length=512)
        model_inputs = tokenizer(table, query, padding="max_length")

        input_ids = model_inputs["input_ids"]
        token_type_ids = np.array(model_inputs["token_type_ids"])
        segment_ids = token_type_ids[:, 0]
        column_ids = token_type_ids[:, 1]
        row_ids = token_type_ids[:, 2]

        expected_results = {'input_ids':[101,2054,2020,1996,6853,3415,1029,102,13433,2015,2053,4062,2136,10876,2051,1013,3394,8370,2685,1015,3590,4754,29267,4765,3771,2136,2447,1005,1055,6584,1015,1024,4466,1024,2340,1012,6185,2509,1015,2570,1016,1015,10391,12022,4226,7895,10625,1013,22996,3868,6584,1009,1014,1012,1022,10819,2015,1016,2459,1017,1017,2703,10555,2136,2447,1005,1055,6584,1009,2654,1012,1020,10819,2015,1017,2403,1018,1023,8709,8183,3126,21351,2078,1010,3781,1012,2136,10958,8865,6584,1009,2871,1012,1022,10819,2015,2410,2260,1019,4090,7986,5292,5677,8151,2771,1011,2990,9187,3868,6584,1009,4413,1012,1015,10819,2015,1020,2184,1020,2322,2030,20282,14262,9035,4754,3868,6584,1009,1015,1024,4002,1012,1016,2184,1022,1021,4868,7918,12023,12023,3868,6584,1009,1015,1024,5890,1012,1018,1019,1020,1022,2260,5261,12436,18116,2137,4382,2136,26447,6584,1009,1015,1024,5890,1012,1022,1022,1019,1023,1021,27339,3995,10125,9711,4906,25101,24657,1011,22033,2386,3868,6564,1009,1015,5001,2321,1018,2184,4583,7986,14383,2075,29488,14906,9351,2971,6564,1009,1015,5001,2340,1017,2340,2676,8527,2014,2696,1052,2243,3868,6564,1009,1015,5001,2260,1016,2260,2861,4575,4477,1011,2128,4710,2137,4382,2136,26447,6564,1009,1015,5001,2459,1015,2410,2539,8963,11503,25457,3022,8512,2522,9654,3868,5594,1009,1016,10876,2324,1014,2403,3943,4074,6415,15204,2072,12496,25378,3868,5594,1009,1016,10876,2403,1014,2321,1018,10704,17921,14906,9351,2971,5594,1009,1016,10876,1023,1014,2385,2340,14915,5795,8512,2522,9654,3868,6640,6228,2539,1014,2459,1016,28328,8945,3126,21351,2015,10625,1013,22996,3868,6255,6228,1018,1014,2324,2321,12270,11956,5232,3868,2260,6228,1021,1014,2539,1019,8473,28027,2080,2474,6371,5232,3868,2184,6228,2385,1014,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'column_ids':[0,0,0,0,0,0,0,0,1,1,2,3,4,5,6,6,6,7,8,1,2,3,3,3,3,4,4,4,4,5,6,6,6,6,6,6,6,6,7,8,1,2,3,3,3,3,4,4,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,4,4,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,3,3,3,3,3,3,4,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,3,3,4,4,4,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,3,3,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,3,4,4,4,4,5,6,6,6,6,6,6,7,8,1,2,3,3,3,3,4,4,4,4,4,4,4,5,6,6,6,7,8,1,2,3,3,3,3,4,4,4,5,6,6,6,7,8,1,2,3,3,3,4,4,4,5,6,6,6,7,8,1,2,3,3,3,3,3,4,4,4,4,5,6,6,6,7,8,1,2,3,3,3,3,4,4,4,4,5,6,6,6,7,8,1,2,3,3,3,3,4,4,4,5,6,6,6,7,8,1,2,3,3,4,4,4,5,6,6,6,7,8,1,2,3,3,4,4,4,4,5,6,7,8,1,2,3,3,3,3,3,4,4,4,4,5,6,7,8,1,2,3,3,4,4,5,6,7,8,1,2,3,3,3,3,3,4,4,5,6,7,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'row_ids':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,19,19,19,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'segment_ids':[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}  # fmt: skip

        self.assertListEqual(input_ids, expected_results["input_ids"])
        self.assertListEqual(segment_ids.tolist(), expected_results["segment_ids"])
        self.assertListEqual(column_ids.tolist(), expected_results["column_ids"])
        self.assertListEqual(row_ids.tolist(), expected_results["row_ids"])

    @unittest.skip(reason="Doesn't support another framework than PyTorch")
    def test_np_encode_plus_sent_to_model(self):
        pass

    @unittest.skip(reason="Chat is not supported")
    def test_chat_template(self):
        pass

    @unittest.skip("Chat is not supported")
    def test_chat_template_return_assistant_tokens_mask(self):
        pass

    @unittest.skip("Chat is not supported")
    def test_chat_template_return_assistant_tokens_mask_truncated(self):
        pass
