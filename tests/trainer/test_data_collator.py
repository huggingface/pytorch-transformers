# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import shutil
import tempfile
import unittest

import numpy as np

from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    DataCollatorWithFlattening,
    DataCollatorWithPadding,
    default_data_collator,
    is_tf_available,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import require_tf, require_torch
from transformers.utils import PaddingStrategy


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


@require_torch
class DataCollatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_default_with_dict(self):
        features = [{"label": i, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue(batch["labels"].equal(torch.tensor(list(range(8)))))
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["inputs"].shape, torch.Size([8, 6]))

        # With label_ids
        features = [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue(batch["labels"].equal(torch.tensor([[0, 1, 2]] * 8)))
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["inputs"].shape, torch.Size([8, 6]))

        # Features can already be tensors
        features = [{"label": i, "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue(batch["labels"].equal(torch.tensor(list(range(8)))))
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["inputs"].shape, torch.Size([8, 10]))

        # Labels can already be tensors
        features = [{"label": torch.tensor(i), "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertTrue(batch["labels"].equal(torch.tensor(list(range(8)))))
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["inputs"].shape, torch.Size([8, 10]))

    def test_default_classification_and_regression(self):
        data_collator = default_data_collator

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": i} for i in range(4)]
        batch = data_collator(features)
        self.assertEqual(batch["labels"].dtype, torch.long)

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": float(i)} for i in range(4)]
        batch = data_collator(features)
        self.assertEqual(batch["labels"].dtype, torch.float)

    def test_default_with_no_labels(self):
        features = [{"label": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, torch.Size([8, 6]))

        # With label_ids
        features = [{"label_ids": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features)
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, torch.Size([8, 6]))

    def test_data_collator_with_padding(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        data_collator = DataCollatorWithPadding(tokenizer)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))

    def test_data_collator_for_token_classification(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": [0, 1, 2], "labels": [0, 1, 2]},
            {"input_ids": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 2, 3, 4, 5]},
        ]

        data_collator = DataCollatorForTokenClassification(tokenizer)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2] + [-100] * 3)

        data_collator = DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=10)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 8]))

        data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2] + [-1] * 3)

        for feature in features:
            feature.pop("labels")

        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

    def test_data_collator_for_token_classification_works_with_pt_tensors(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([0, 1, 2])},
            {"input_ids": torch.tensor([0, 1, 2, 3, 4, 5]), "labels": torch.tensor([0, 1, 2, 3, 4, 5])},
        ]

        data_collator = DataCollatorForTokenClassification(tokenizer)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2] + [-100] * 3)

        data_collator = DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=10)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 8]))

        data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2] + [-1] * 3)

        for feature in features:
            feature.pop("labels")

        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

    def _test_data_collator_for_seq2seq(self, to_torch):
        def create_features(to_torch):
            if to_torch:
                features = [
                    {"input_ids": torch.tensor(list(range(3))), "labels": torch.tensor(list(range(3)))},
                    {"input_ids": torch.tensor(list(range(6))), "labels": torch.tensor(list(range(6)))},
                ]
            else:
                features = [
                    {"input_ids": list(range(3)), "labels": list(range(3))},
                    {"input_ids": list(range(6)), "labels": list(range(6))},
                ]
            return features

        tokenizer = BertTokenizer(self.vocab_file)
        features = create_features(to_torch)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["input_ids"][1].tolist(), list(range(6)))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)) + [-100] * 3)
        self.assertEqual(batch["labels"][1].tolist(), list(range(6)))

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=7)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 7]))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 4)
        self.assertEqual(batch["input_ids"][1].tolist(), list(range(6)) + [tokenizer.pad_token_id] * 1)
        self.assertEqual(batch["labels"].shape, torch.Size([2, 7]))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)) + [-100] * 4)
        self.assertEqual(batch["labels"][1].tolist(), list(range(6)) + [-100] * 1)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.DO_NOT_PAD)
        with self.assertRaises(ValueError):
            # expects an error due to unequal shapes to create tensor
            data_collator(features)
        batch = data_collator([features[0], features[0]])
        input_ids = features[0]["input_ids"] if not to_torch else features[0]["input_ids"].tolist()
        labels = features[0]["labels"] if not to_torch else features[0]["labels"].tolist()
        self.assertEqual(batch["input_ids"][0].tolist(), input_ids)
        self.assertEqual(batch["input_ids"][1].tolist(), input_ids)
        self.assertEqual(batch["labels"][0].tolist(), labels)
        self.assertEqual(batch["labels"][1].tolist(), labels)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 8]))

        # side effects on labels cause mismatch on longest strategy
        features = create_features(to_torch)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1)
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["input_ids"][1].tolist(), list(range(6)))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)) + [-1] * 3)
        self.assertEqual(batch["labels"][1].tolist(), list(range(6)))

        for feature in features:
            feature.pop("labels")

        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)

    def test_data_collator_for_seq2seq_with_lists(self):
        self._test_data_collator_for_seq2seq(to_torch=False)

    def test_data_collator_for_seq2seq_with_pt(self):
        self._test_data_collator_for_seq2seq(to_torch=True)

    def _test_no_pad_and_pad(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        tokenizer.pad_token = None
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        with self.assertRaises(ValueError):
            # Expect error due to padding token missing
            data_collator(pad_features)

        set_seed(42)  # For reproducibility
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

    def test_data_collator_for_language_modeling(self):
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

        no_pad_features = [list(range(10)), list(range(10))]
        pad_features = [list(range(5)), list(range(10))]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

    def test_data_collator_for_whole_word_mask(self):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="pt")

        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        # Features can already be tensors
        features = [{"input_ids": np.arange(10)}, {"input_ids": np.arange(10)}]
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

    def test_plm(self):
        tokenizer = BertTokenizer(self.vocab_file)
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]

        data_collator = DataCollatorForPermutationLanguageModeling(tokenizer)

        batch = data_collator(pad_features)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["perm_mask"].shape, torch.Size((2, 10, 10)))
        self.assertEqual(batch["target_mapping"].shape, torch.Size((2, 10, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        batch = data_collator(no_pad_features)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["perm_mask"].shape, torch.Size((2, 10, 10)))
        self.assertEqual(batch["target_mapping"].shape, torch.Size((2, 10, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        example = [np.random.randint(0, 5, [5])]
        with self.assertRaises(ValueError):
            # Expect error due to odd sequence length
            data_collator(example)

    def test_nsp(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 5)))
        self.assertEqual(batch["token_type_ids"].shape, torch.Size((2, 5)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 5)))
        self.assertEqual(batch["next_sentence_label"].shape, torch.Size((2,)))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 8)))
        self.assertEqual(batch["token_type_ids"].shape, torch.Size((2, 8)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 8)))
        self.assertEqual(batch["next_sentence_label"].shape, torch.Size((2,)))

    def test_sop(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {
                "input_ids": torch.tensor([0, 1, 2, 3, 4]),
                "token_type_ids": torch.tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 5)))
        self.assertEqual(batch["token_type_ids"].shape, torch.Size((2, 5)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 5)))
        self.assertEqual(batch["sentence_order_label"].shape, torch.Size((2,)))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 8)))
        self.assertEqual(batch["token_type_ids"].shape, torch.Size((2, 8)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 8)))
        self.assertEqual(batch["sentence_order_label"].shape, torch.Size((2,)))


@require_torch
class DataCollatorImmutabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def _turn_to_none(self, item):
        """used to convert `item` to `None` type"""
        return None

    def _validate_original_data_against_collated_data(self, collator, original_data, batch_data):
        # we only care about side effects, the results are tested elsewhere
        collator(batch_data)

        # we go through every item and convert to `primitive` datatypes if necessary
        # then compares for equivalence for the original data and the data that has been passed through the collator
        for original, batch in zip(original_data, batch_data):
            for original_val, batch_val in zip(original.values(), batch.values()):
                if isinstance(original_val, (np.ndarray, torch.Tensor)):
                    self.assertEqual(original_val.tolist(), batch_val.tolist())
                else:
                    self.assertEqual(original_val, batch_val)

    def _validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
        self, collator, base_data, input_key, input_datatype, label_key, label_datatype, ignore_label=False
    ):
        # using the arguments to recreate the features with their respective (potentially new) datatypes
        features_original = [
            {label_key: label_datatype(sample[label_key]), input_key: input_datatype(sample[input_key])}
            for sample in base_data
        ]
        features_batch = [
            {label_key: label_datatype(sample[label_key]), input_key: input_datatype(sample[input_key])}
            for sample in base_data
        ]

        # some collators do not use labels, or sometimes we want to check if the collator with labels can handle such cases
        if ignore_label:
            for original, batch in zip(features_original, features_batch):
                original.pop(label_key)
                batch.pop(label_key)

        self._validate_original_data_against_collated_data(
            collator=collator, original_data=features_original, batch_data=features_batch
        )

    def test_default_collator_immutability(self):
        features_base_single_label = [{"label": i, "inputs": (0, 1, 2, 3, 4, 5)} for i in range(4)]
        features_base_multiple_labels = [{"label": (0, 1, 2), "inputs": (0, 1, 2, 3, 4, 5)} for i in range(4)]

        for datatype_input, datatype_label in [
            (list, int),
            (list, float),
            (np.array, int),
            (np.array, torch.tensor),
            (list, self._turn_to_none),
        ]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=default_data_collator,
                base_data=features_base_single_label,
                input_key="inputs",
                input_datatype=datatype_input,
                label_key="label",
                label_datatype=datatype_label,
            )

        for datatype_input, datatype_label in [(list, list), (list, self._turn_to_none)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=default_data_collator,
                base_data=features_base_multiple_labels,
                input_key="inputs",
                input_datatype=datatype_input,
                label_key="label",
                label_datatype=datatype_label,
            )

        features_base_single_label_alt = [{"input_ids": (0, 1, 2, 3, 4), "label": float(i)} for i in range(4)]
        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=default_data_collator,
            base_data=features_base_single_label_alt,
            input_key="input_ids",
            input_datatype=list,
            label_key="label",
            label_datatype=float,
        )

    def test_with_padding_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]
        features_batch = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10)
        self._validate_original_data_against_collated_data(
            collator=data_collator, original_data=features_original, batch_data=features_batch
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        self._validate_original_data_against_collated_data(
            collator=data_collator, original_data=features_original, batch_data=features_batch
        )

    def test_for_token_classification_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": (0, 1, 2), "labels": (0, 1, 2)},
            {"input_ids": (0, 1, 2, 3, 4, 5), "labels": (0, 1, 2, 3, 4, 5)},
        ]
        token_classification_collators = [
            DataCollatorForTokenClassification(tokenizer),
            DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=10),
            DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
            DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1),
        ]

        for datatype_input, datatype_label in [(list, list), (torch.tensor, torch.tensor)]:
            for collator in token_classification_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                )

        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=token_classification_collators[-1],
            base_data=features_base,
            input_key="input_ids",
            input_datatype=datatype_input,
            label_key="labels",
            label_datatype=datatype_label,
            ignore_label=True,
        )

    def test_seq2seq_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(6)), "labels": list(range(6))},
        ]
        seq2seq_collators = [
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST),
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=7),
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8),
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1),
        ]

        for datatype_input, datatype_label in [(list, list), (torch.tensor, torch.tensor)]:
            for collator in seq2seq_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                )

        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=seq2seq_collators[-1],
            base_data=features_base,
            input_key="input_ids",
            input_datatype=datatype_input,
            label_key="labels",
            label_datatype=datatype_label,
            ignore_label=True,
        )

        features_base_no_pad = [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(3)), "labels": list(range(3))},
        ]
        seq2seq_no_padding_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.DO_NOT_PAD)
        for datatype_input, datatype_label in [(list, list), (torch.tensor, torch.tensor)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=seq2seq_no_padding_collator,
                base_data=features_base_no_pad,
                input_key="input_ids",
                input_datatype=datatype_input,
                label_key="labels",
                label_datatype=datatype_label,
            )

    def test_language_modelling_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base_no_pad = [
            {"input_ids": tuple(range(10)), "labels": (1,)},
            {"input_ids": tuple(range(10)), "labels": (1,)},
        ]
        features_base_pad = [
            {"input_ids": tuple(range(5)), "labels": (1,)},
            {"input_ids": tuple(range(5)), "labels": (1,)},
        ]
        lm_collators = [
            DataCollatorForLanguageModeling(tokenizer, mlm=False),
            DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
            DataCollatorForLanguageModeling(tokenizer),
            DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8),
        ]

        for datatype_input, datatype_label in [(list, list), (torch.tensor, torch.tensor)]:
            for collator in lm_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base_no_pad,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                    ignore_label=True,
                )

                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base_pad,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                    ignore_label=True,
                )

    def test_whole_world_masking_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": list(range(10)), "labels": (1,)},
            {"input_ids": list(range(10)), "labels": (1,)},
        ]
        whole_word_masking_collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="pt")

        for datatype_input, datatype_label in [(list, list), (np.array, np.array)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=whole_word_masking_collator,
                base_data=features_base,
                input_key="input_ids",
                input_datatype=datatype_input,
                label_key="labels",
                label_datatype=datatype_label,
                ignore_label=True,
            )

    def test_permutation_language_modelling_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        plm_collator = DataCollatorForPermutationLanguageModeling(tokenizer)

        no_pad_features_original = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        no_pad_features_batch = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        self._validate_original_data_against_collated_data(
            collator=plm_collator, original_data=no_pad_features_original, batch_data=no_pad_features_batch
        )

        pad_features_original = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        pad_features_batch = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._validate_original_data_against_collated_data(
            collator=plm_collator, original_data=pad_features_original, batch_data=pad_features_batch
        )

    def test_next_sentence_prediction_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]
        features_batch = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]

        nsp_collator = DataCollatorForLanguageModeling(tokenizer)
        self._validate_original_data_against_collated_data(
            collator=nsp_collator, original_data=features_original, batch_data=features_batch
        )

        nsp_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        self._validate_original_data_against_collated_data(
            collator=nsp_collator, original_data=features_original, batch_data=features_batch
        )

    def test_sentence_order_prediction_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [
            {
                "input_ids": torch.tensor([0, 1, 2, 3, 4]),
                "token_type_ids": torch.tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]
        features_batch = [
            {
                "input_ids": torch.tensor([0, 1, 2, 3, 4]),
                "token_type_ids": torch.tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]

        sop_collator = DataCollatorForLanguageModeling(tokenizer)
        self._validate_original_data_against_collated_data(
            collator=sop_collator, original_data=features_original, batch_data=features_batch
        )

        sop_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        self._validate_original_data_against_collated_data(
            collator=sop_collator, original_data=features_original, batch_data=features_batch
        )


@require_tf
class TFDataCollatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_default_with_dict(self):
        features = [{"label": i, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].numpy().tolist(), list(range(8)))
        self.assertEqual(batch["labels"].dtype, tf.int64)
        self.assertEqual(batch["inputs"].shape.as_list(), [8, 6])

        # With label_ids
        features = [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].numpy().tolist(), ([[0, 1, 2]] * 8))
        self.assertEqual(batch["labels"].dtype, tf.int64)
        self.assertEqual(batch["inputs"].shape.as_list(), [8, 6])

        # Features can already be tensors
        features = [{"label": i, "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].numpy().tolist(), (list(range(8))))
        self.assertEqual(batch["labels"].dtype, tf.int64)
        self.assertEqual(batch["inputs"].shape.as_list(), [8, 10])

        # Labels can already be tensors
        features = [{"label": np.array(i), "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].dtype, tf.int64)
        self.assertEqual(batch["labels"].numpy().tolist(), list(range(8)))
        self.assertEqual(batch["labels"].dtype, tf.int64)
        self.assertEqual(batch["inputs"].shape.as_list(), [8, 10])

    def test_numpy_dtype_preservation(self):
        data_collator = default_data_collator

        # Confirms that numpy inputs are handled correctly even when scalars
        features = [{"input_ids": np.array([0, 1, 2, 3, 4]), "label": np.int64(i)} for i in range(4)]
        batch = data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].dtype, tf.int64)

    def test_default_classification_and_regression(self):
        data_collator = default_data_collator

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": i} for i in range(4)]
        batch = data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].dtype, tf.int64)

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": float(i)} for i in range(4)]
        batch = data_collator(features, return_tensors="tf")
        self.assertEqual(batch["labels"].dtype, tf.float32)

    def test_default_with_no_labels(self):
        features = [{"label": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="tf")
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape.as_list(), [8, 6])

        # With label_ids
        features = [{"label_ids": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="tf")
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape.as_list(), [8, 6])

    def test_data_collator_with_padding(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 6])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, [2, 8])

    def test_data_collator_for_token_classification(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": [0, 1, 2], "labels": [0, 1, 2]},
            {"input_ids": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 2, 3, 4, 5]},
        ]

        data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 6])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape.as_list(), [2, 6])
        self.assertEqual(batch["labels"][0].numpy().tolist(), [0, 1, 2] + [-100] * 3)

        data_collator = DataCollatorForTokenClassification(
            tokenizer, padding="max_length", max_length=10, return_tensors="tf"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 8])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 8])

        data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 6])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape.as_list(), [2, 6])
        self.assertEqual(batch["labels"][0].numpy().tolist(), [0, 1, 2] + [-1] * 3)

    def test_data_collator_for_seq2seq(self):
        def create_features():
            return [
                {"input_ids": list(range(3)), "labels": list(range(3))},
                {"input_ids": list(range(6)), "labels": list(range(6))},
            ]

        tokenizer = BertTokenizer(self.vocab_file)
        features = create_features()

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, return_tensors="tf")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 6])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["input_ids"][1].numpy().tolist(), list(range(6)))
        self.assertEqual(batch["labels"].shape.as_list(), [2, 6])
        self.assertEqual(batch["labels"][0].numpy().tolist(), list(range(3)) + [-100] * 3)
        self.assertEqual(batch["labels"][1].numpy().tolist(), list(range(6)))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=7, return_tensors="tf"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 7])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), list(range(3)) + [tokenizer.pad_token_id] * 4)
        self.assertEqual(batch["input_ids"][1].numpy().tolist(), list(range(6)) + [tokenizer.pad_token_id] * 1)
        self.assertEqual(batch["labels"].shape.as_list(), [2, 7])
        self.assertEqual(batch["labels"][0].numpy().tolist(), list(range(3)) + [-100] * 4)
        self.assertEqual(batch["labels"][1].numpy().tolist(), list(range(6)) + [-100] * 1)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.DO_NOT_PAD, return_tensors="tf")
        with self.assertRaises(ValueError):
            # expects an error due to unequal shapes to create tensor
            data_collator(features)
        batch = data_collator([features[0], features[0]])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), features[0]["input_ids"])
        self.assertEqual(batch["input_ids"][1].numpy().tolist(), features[0]["input_ids"])
        self.assertEqual(batch["labels"][0].numpy().tolist(), features[0]["labels"])
        self.assertEqual(batch["labels"][1].numpy().tolist(), features[0]["labels"])

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8, return_tensors="tf"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 8])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 8])

        # side effects on labels cause mismatch on longest strategy
        features = create_features()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1, return_tensors="tf"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 6])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["input_ids"][1].numpy().tolist(), list(range(6)))
        self.assertEqual(batch["labels"].shape.as_list(), [2, 6])
        self.assertEqual(batch["labels"][0].numpy().tolist(), list(range(3)) + [-1] * 3)
        self.assertEqual(batch["labels"][1].numpy().tolist(), list(range(6)))

        for feature in features:
            feature.pop("labels")

        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 6])
        self.assertEqual(batch["input_ids"][0].numpy().tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)

    def _test_no_pad_and_pad(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="tf"
        )
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 16])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 16])

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 16])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 16])

        tokenizer.pad_token = None
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")
        with self.assertRaises(ValueError):
            # Expect error due to padding token missing
            data_collator(pad_features)

        set_seed(42)  # For reproducibility
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="tf")
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(tf.reduce_any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"].numpy()[~masked_tokens.numpy()].tolist()))

        batch = data_collator(pad_features, return_tensors="tf")
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(tf.reduce_any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"].numpy()[~masked_tokens.numpy()].tolist()))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 16])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 16])

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(tf.reduce_any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"].numpy()[~masked_tokens.numpy()].tolist()))

        batch = data_collator(pad_features, return_tensors="tf")
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 16])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 16])

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(tf.reduce_any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"].numpy()[~masked_tokens.numpy()].tolist()))

    def test_probability_sum_error(self):
        """Test that the sum of mask_replace_prob and random_replace_prob exceeding 1 raises an error."""
        tokenizer = BertTokenizer(self.vocab_file)
        with self.assertRaises(ValueError):
            DataCollatorForLanguageModeling(tokenizer=tokenizer, mask_replace_prob=0.9, random_replace_prob=0.2)

    def test_all_mask_replacement(self):
        """Test behavior when mask_replace_prob=1."""
        tokenizer = BertTokenizer(self.vocab_file)

        # pytorch call
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mask_replace_prob=1, random_replace_prob=0, return_tensors="pt"
        )

        inputs = torch.tensor([0, 1, 2, 3, 4, 5])
        features = [{"input_ids": inputs} for _ in range(8)]
        batch = collator(features)

        # confirm that every token is either the original token or [MASK]
        self.assertTrue(torch.all((batch["input_ids"] == inputs) | (batch["input_ids"] == tokenizer.mask_token_id)))

        # tf call
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mask_replace_prob=1, random_replace_prob=0, return_tensors="tf"
        )
        inputs = tf.constant([0, 1, 2, 3, 4, 5])
        features = [{"input_ids": inputs} for _ in range(8)]
        batch = collator(features)

        # confirm that every token is either the original token or [MASK]
        self.assertTrue(
            tf.reduce_all((batch["input_ids"] == inputs) | (batch["input_ids"] == tokenizer.mask_token_id))
        )

        # numpy call
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mask_replace_prob=1, random_replace_prob=0, return_tensors="np"
        )
        inputs = np.array([0, 1, 2, 3, 4, 5])
        features = [{"input_ids": inputs} for _ in range(8)]
        batch = collator(features)

        # confirm that every token is either the original token or [MASK]
        self.assertTrue(np.all((batch["input_ids"] == inputs) | (batch["input_ids"] == tokenizer.mask_token_id)))

    def test_data_collator_for_language_modeling(self):
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

        no_pad_features = [list(range(10)), list(range(10))]
        pad_features = [list(range(5)), list(range(10))]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

    def test_data_collator_for_whole_word_mask(self):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="tf")

        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        # Features can already be tensors
        features = [{"input_ids": np.arange(10)}, {"input_ids": np.arange(10)}]
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

    def test_plm(self):
        tokenizer = BertTokenizer(self.vocab_file)
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]

        data_collator = DataCollatorForPermutationLanguageModeling(tokenizer, return_tensors="tf")

        batch = data_collator(pad_features)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["perm_mask"].shape.as_list(), [2, 10, 10])
        self.assertEqual(batch["target_mapping"].shape.as_list(), [2, 10, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        batch = data_collator(no_pad_features)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 10])
        self.assertEqual(batch["perm_mask"].shape.as_list(), [2, 10, 10])
        self.assertEqual(batch["target_mapping"].shape.as_list(), [2, 10, 10])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 10])

        example = [np.random.randint(0, 5, [5])]
        with self.assertRaises(ValueError):
            # Expect error due to odd sequence length
            data_collator(example)

    def test_nsp(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]
        data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="tf")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 5])
        self.assertEqual(batch["token_type_ids"].shape.as_list(), [2, 5])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 5])
        self.assertEqual(batch["next_sentence_label"].shape.as_list(), [2])

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 8])
        self.assertEqual(batch["token_type_ids"].shape.as_list(), [2, 8])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 8])
        self.assertEqual(batch["next_sentence_label"].shape.as_list(), [2])

    def test_sop(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {
                "input_ids": tf.convert_to_tensor([0, 1, 2, 3, 4]),
                "token_type_ids": tf.convert_to_tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]
        data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="tf")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 5])
        self.assertEqual(batch["token_type_ids"].shape.as_list(), [2, 5])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 5])
        self.assertEqual(batch["sentence_order_label"].shape.as_list(), [2])

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape.as_list(), [2, 8])
        self.assertEqual(batch["token_type_ids"].shape.as_list(), [2, 8])
        self.assertEqual(batch["labels"].shape.as_list(), [2, 8])
        self.assertEqual(batch["sentence_order_label"].shape.as_list(), [2])


@require_tf
class TFDataCollatorImmutabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def _turn_to_none(self, item):
        """used to convert `item` to `None` type"""
        return None

    def _validate_original_data_against_collated_data(self, collator, original_data, batch_data):
        # we only care about side effects, the results are tested elsewhere
        collator(batch_data)

        # we go through every item and convert to `primitive` datatypes if necessary
        # then compares for equivalence for the original data and the data that has been passed through the collator
        for original, batch in zip(original_data, batch_data):
            for original_val, batch_val in zip(original.values(), batch.values()):
                if isinstance(original_val, np.ndarray):
                    self.assertEqual(original_val.tolist(), batch_val.tolist())
                elif isinstance(original_val, tf.Tensor):
                    self.assertEqual(original_val.numpy().tolist(), batch_val.numpy().tolist())
                else:
                    self.assertEqual(original_val, batch_val)

    def _validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
        self, collator, base_data, input_key, input_datatype, label_key, label_datatype, ignore_label=False
    ):
        # using the arguments to recreate the features with their respective (potentially new) datatypes
        features_original = [
            {label_key: label_datatype(sample[label_key]), input_key: input_datatype(sample[input_key])}
            for sample in base_data
        ]
        features_batch = [
            {label_key: label_datatype(sample[label_key]), input_key: input_datatype(sample[input_key])}
            for sample in base_data
        ]

        # some collators do not use labels, or sometimes we want to check if the collator with labels can handle such cases
        if ignore_label:
            for original, batch in zip(features_original, features_batch):
                original.pop(label_key)
                batch.pop(label_key)

        self._validate_original_data_against_collated_data(
            collator=collator, original_data=features_original, batch_data=features_batch
        )

    def test_default_collator_immutability(self):
        features_base_single_label = [{"label": i, "inputs": (0, 1, 2, 3, 4, 5)} for i in range(4)]
        features_base_multiple_labels = [{"label": (0, 1, 2), "inputs": (0, 1, 2, 3, 4, 5)} for i in range(4)]

        for datatype_input, datatype_label in [
            (list, int),
            (list, float),
            (np.array, int),
            (np.array, tf.constant),
            (list, self._turn_to_none),
        ]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=lambda x: default_data_collator(x, return_tensors="tf"),
                base_data=features_base_single_label,
                input_key="inputs",
                input_datatype=datatype_input,
                label_key="label",
                label_datatype=datatype_label,
            )

        for datatype_input, datatype_label in [(list, list), (list, self._turn_to_none)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=lambda x: default_data_collator(x, return_tensors="tf"),
                base_data=features_base_multiple_labels,
                input_key="inputs",
                input_datatype=datatype_input,
                label_key="label",
                label_datatype=datatype_label,
            )

        features_base_single_label_alt = [{"input_ids": (0, 1, 2, 3, 4), "label": float(i)} for i in range(4)]
        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=lambda x: default_data_collator(x, return_tensors="tf"),
            base_data=features_base_single_label_alt,
            input_key="input_ids",
            input_datatype=list,
            label_key="label",
            label_datatype=float,
        )

    def test_with_padding_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]
        features_batch = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10, return_tensors="tf")
        self._validate_original_data_against_collated_data(
            collator=data_collator, original_data=features_original, batch_data=features_batch
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        self._validate_original_data_against_collated_data(
            collator=data_collator, original_data=features_original, batch_data=features_batch
        )

    def test_for_token_classification_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": (0, 1, 2), "labels": (0, 1, 2)},
            {"input_ids": (0, 1, 2, 3, 4, 5), "labels": (0, 1, 2, 3, 4, 5)},
        ]
        token_classification_collators = [
            DataCollatorForTokenClassification(tokenizer, return_tensors="tf"),
            DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=10, return_tensors="tf"),
            DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8, return_tensors="tf"),
            DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1, return_tensors="tf"),
        ]

        for datatype_input, datatype_label in [(list, list)]:
            for collator in token_classification_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                )

        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=token_classification_collators[-1],
            base_data=features_base,
            input_key="input_ids",
            input_datatype=datatype_input,
            label_key="labels",
            label_datatype=datatype_label,
            ignore_label=True,
        )

    def test_seq2seq_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(6)), "labels": list(range(6))},
        ]
        seq2seq_collators = [
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, return_tensors="tf"),
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=7, return_tensors="tf"),
            DataCollatorForSeq2Seq(
                tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8, return_tensors="tf"
            ),
            DataCollatorForSeq2Seq(
                tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1, return_tensors="tf"
            ),
        ]

        for datatype_input, datatype_label in [(list, list)]:
            for collator in seq2seq_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                )

        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=seq2seq_collators[-1],
            base_data=features_base,
            input_key="input_ids",
            input_datatype=datatype_input,
            label_key="labels",
            label_datatype=datatype_label,
            ignore_label=True,
        )

        features_base_no_pad = [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(3)), "labels": list(range(3))},
        ]
        seq2seq_no_padding_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.DO_NOT_PAD, return_tensors="tf"
        )
        for datatype_input, datatype_label in [(list, list)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=seq2seq_no_padding_collator,
                base_data=features_base_no_pad,
                input_key="input_ids",
                input_datatype=datatype_input,
                label_key="labels",
                label_datatype=datatype_label,
            )

    def test_language_modelling_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base_no_pad = [
            {"input_ids": tuple(range(10)), "labels": (1,)},
            {"input_ids": tuple(range(10)), "labels": (1,)},
        ]
        features_base_pad = [
            {"input_ids": tuple(range(5)), "labels": (1,)},
            {"input_ids": tuple(range(5)), "labels": (1,)},
        ]
        lm_collators = [
            DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf"),
            DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="tf"),
            DataCollatorForLanguageModeling(tokenizer, return_tensors="tf"),
            DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="tf"),
        ]

        for datatype_input, datatype_label in [(list, list)]:
            for collator in lm_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base_no_pad,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                    ignore_label=True,
                )

                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base_pad,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                    ignore_label=True,
                )

    def test_whole_world_masking_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": list(range(10)), "labels": (1,)},
            {"input_ids": list(range(10)), "labels": (1,)},
        ]
        whole_word_masking_collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="tf")

        for datatype_input, datatype_label in [(list, list), (np.array, np.array)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=whole_word_masking_collator,
                base_data=features_base,
                input_key="input_ids",
                input_datatype=datatype_input,
                label_key="labels",
                label_datatype=datatype_label,
                ignore_label=True,
            )

    def test_permutation_language_modelling_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        plm_collator = DataCollatorForPermutationLanguageModeling(tokenizer, return_tensors="tf")

        no_pad_features_original = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        no_pad_features_batch = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        self._validate_original_data_against_collated_data(
            collator=plm_collator, original_data=no_pad_features_original, batch_data=no_pad_features_batch
        )

        pad_features_original = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        pad_features_batch = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._validate_original_data_against_collated_data(
            collator=plm_collator, original_data=pad_features_original, batch_data=pad_features_batch
        )

    def test_next_sentence_prediction_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]
        features_batch = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]

        nsp_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="tf")
        self._validate_original_data_against_collated_data(
            collator=nsp_collator, original_data=features_original, batch_data=features_batch
        )

        nsp_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        self._validate_original_data_against_collated_data(
            collator=nsp_collator, original_data=features_original, batch_data=features_batch
        )

    def test_sentence_order_prediction_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [
            {
                "input_ids": tf.convert_to_tensor([0, 1, 2, 3, 4]),
                "token_type_ids": tf.convert_to_tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]
        features_batch = [
            {
                "input_ids": tf.convert_to_tensor([0, 1, 2, 3, 4]),
                "token_type_ids": tf.convert_to_tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]

        sop_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="tf")
        self._validate_original_data_against_collated_data(
            collator=sop_collator, original_data=features_original, batch_data=features_batch
        )

        sop_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="tf")
        self._validate_original_data_against_collated_data(
            collator=sop_collator, original_data=features_original, batch_data=features_batch
        )


class NumpyDataCollatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_default_with_dict(self):
        features = [{"label": i, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].tolist(), list(range(8)))
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(batch["inputs"].shape, (8, 6))

        # With label_ids
        features = [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].tolist(), [[0, 1, 2]] * 8)
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(batch["inputs"].shape, (8, 6))

        # Features can already be tensors
        features = [{"label": i, "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].tolist(), list(range(8)))
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(batch["inputs"].shape, (8, 10))

        # Labels can already be tensors
        features = [{"label": np.array(i), "inputs": np.random.randint(0, 10, [10])} for i in range(8)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(batch["labels"].tolist(), (list(range(8))))
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(batch["inputs"].shape, (8, 10))

    def test_default_classification_and_regression(self):
        data_collator = default_data_collator

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": i} for i in range(4)]
        batch = data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].dtype, np.int64)

        features = [{"input_ids": [0, 1, 2, 3, 4], "label": float(i)} for i in range(4)]
        batch = data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].dtype, np.float32)

    def test_default_with_no_labels(self):
        features = [{"label": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, (8, 6))

        # With label_ids
        features = [{"label_ids": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, (8, 6))

    def test_data_collator_with_padding(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 8))

    def test_data_collator_with_flattening(self):
        features = [
            {"input_ids": [10, 11, 12]},
            {"input_ids": [20, 21, 22, 23, 24, 25]},
            {"input_ids": [30, 31, 32, 33, 34, 35, 36]},
        ]

        data_collator = DataCollatorWithFlattening(return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (1, 16))
        self.assertEqual(
            batch["input_ids"][0].tolist(), [10, 11, 12, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36]
        )
        self.assertNotIn("attention_mask", batch)
        self.assertIn("position_ids", batch)
        self.assertEqual(batch["position_ids"].shape, (1, 16))
        self.assertEqual(batch["position_ids"][0].tolist(), [0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6])

    def test_data_collator_for_token_classification(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": [0, 1, 2], "labels": [0, 1, 2]},
            {"input_ids": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 2, 3, 4, 5]},
        ]

        data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, (2, 6))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2] + [-100] * 3)

        data_collator = DataCollatorForTokenClassification(
            tokenizer, padding="max_length", max_length=10, return_tensors="np"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertEqual(batch["labels"].shape, (2, 8))

        data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, (2, 6))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2] + [-1] * 3)

    def test_data_collator_for_seq2seq(self):
        def create_features():
            return [
                {"input_ids": list(range(3)), "labels": list(range(3))},
                {"input_ids": list(range(6)), "labels": list(range(6))},
            ]

        tokenizer = BertTokenizer(self.vocab_file)
        features = create_features()

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, return_tensors="np")
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["input_ids"][1].tolist(), list(range(6)))
        self.assertEqual(batch["labels"].shape, (2, 6))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)) + [-100] * 3)
        self.assertEqual(batch["labels"][1].tolist(), list(range(6)))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=7, return_tensors="np"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 7))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 4)
        self.assertEqual(batch["input_ids"][1].tolist(), list(range(6)) + [tokenizer.pad_token_id] * 1)
        self.assertEqual(batch["labels"].shape, (2, 7))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)) + [-100] * 4)
        self.assertEqual(batch["labels"][1].tolist(), list(range(6)) + [-100] * 1)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.DO_NOT_PAD, return_tensors="np")
        # numpy doesn't have issues handling unequal shapes via `dtype=object`
        # with self.assertRaises(ValueError):
        #     data_collator(features)
        batch = data_collator([features[0], features[0]])
        self.assertEqual(batch["input_ids"][0].tolist(), features[0]["input_ids"])
        self.assertEqual(batch["input_ids"][1].tolist(), features[0]["input_ids"])
        self.assertEqual(batch["labels"][0].tolist(), features[0]["labels"])
        self.assertEqual(batch["labels"][1].tolist(), features[0]["labels"])

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8, return_tensors="np"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertEqual(batch["labels"].shape, (2, 8))

        # side effects on labels cause mismatch on longest strategy
        features = create_features()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1, return_tensors="np"
        )
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["input_ids"][1].tolist(), list(range(6)))
        self.assertEqual(batch["labels"].shape, (2, 6))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)) + [-1] * 3)
        self.assertEqual(batch["labels"][1].tolist(), list(range(6)))

        for feature in features:
            feature.pop("labels")

        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)) + [tokenizer.pad_token_id] * 3)

    def _test_no_pad_and_pad(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="np")
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        batch = data_collator(pad_features, return_tensors="np")
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="np"
        )
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, (2, 16))
        self.assertEqual(batch["labels"].shape, (2, 16))

        batch = data_collator(pad_features, return_tensors="np")
        self.assertEqual(batch["input_ids"].shape, (2, 16))
        self.assertEqual(batch["labels"].shape, (2, 16))

        tokenizer.pad_token = None
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="np")
        with self.assertRaises(ValueError):
            # Expect error due to padding token missing
            data_collator(pad_features)

        set_seed(42)  # For reproducibility
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(np.any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(np.any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, (2, 16))
        self.assertEqual(batch["labels"].shape, (2, 16))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(np.any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, (2, 16))
        self.assertEqual(batch["labels"].shape, (2, 16))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(np.any(masked_tokens))
        # self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

    def test_data_collator_for_language_modeling(self):
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

        no_pad_features = [list(range(10)), list(range(10))]
        pad_features = [list(range(5)), list(range(10))]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

    def test_data_collator_for_whole_word_mask(self):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="np")

        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        # Features can already be tensors
        features = [{"input_ids": np.arange(10)}, {"input_ids": np.arange(10)}]
        batch = data_collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

    def test_plm(self):
        tokenizer = BertTokenizer(self.vocab_file)
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]

        data_collator = DataCollatorForPermutationLanguageModeling(tokenizer, return_tensors="np")

        batch = data_collator(pad_features)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["perm_mask"].shape, (2, 10, 10))
        self.assertEqual(batch["target_mapping"].shape, (2, 10, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        batch = data_collator(no_pad_features)
        self.assertIsInstance(batch, dict)
        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["perm_mask"].shape, (2, 10, 10))
        self.assertEqual(batch["target_mapping"].shape, (2, 10, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

        example = [np.random.randint(0, 5, [5])]
        with self.assertRaises(ValueError):
            # Expect error due to odd sequence length
            data_collator(example)

    def test_nsp(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]
        data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["token_type_ids"].shape, (2, 5))
        self.assertEqual(batch["labels"].shape, (2, 5))
        self.assertEqual(batch["next_sentence_label"].shape, (2,))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertEqual(batch["token_type_ids"].shape, (2, 8))
        self.assertEqual(batch["labels"].shape, (2, 8))
        self.assertEqual(batch["next_sentence_label"].shape, (2,))

    def test_sop(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {
                "input_ids": np.array([0, 1, 2, 3, 4]),
                "token_type_ids": np.array([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]
        data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["token_type_ids"].shape, (2, 5))
        self.assertEqual(batch["labels"].shape, (2, 5))
        self.assertEqual(batch["sentence_order_label"].shape, (2,))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertEqual(batch["token_type_ids"].shape, (2, 8))
        self.assertEqual(batch["labels"].shape, (2, 8))
        self.assertEqual(batch["sentence_order_label"].shape, (2,))


class NumpyDataCollatorImmutabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def _turn_to_none(self, item):
        """used to convert `item` to `None` type"""
        return None

    def _validate_original_data_against_collated_data(self, collator, original_data, batch_data):
        # we only care about side effects, the results are tested elsewhere
        collator(batch_data)

        # we go through every item and convert to `primitive` datatypes if necessary
        # then compares for equivalence for the original data and the data that has been passed through the collator
        for original, batch in zip(original_data, batch_data):
            for original_val, batch_val in zip(original.values(), batch.values()):
                if isinstance(original_val, np.ndarray):
                    self.assertEqual(original_val.tolist(), batch_val.tolist())
                else:
                    self.assertEqual(original_val, batch_val)

    def _validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
        self, collator, base_data, input_key, input_datatype, label_key, label_datatype, ignore_label=False
    ):
        # using the arguments to recreate the features with their respective (potentially new) datatypes
        features_original = [
            {label_key: label_datatype(sample[label_key]), input_key: input_datatype(sample[input_key])}
            for sample in base_data
        ]
        features_batch = [
            {label_key: label_datatype(sample[label_key]), input_key: input_datatype(sample[input_key])}
            for sample in base_data
        ]

        # some collators do not use labels, or sometimes we want to check if the collator with labels can handle such cases
        if ignore_label:
            for original, batch in zip(features_original, features_batch):
                original.pop(label_key)
                batch.pop(label_key)

        self._validate_original_data_against_collated_data(
            collator=collator, original_data=features_original, batch_data=features_batch
        )

    def test_default_collator_immutability(self):
        features_base_single_label = [{"label": i, "inputs": (0, 1, 2, 3, 4, 5)} for i in range(4)]
        features_base_multiple_labels = [{"label": (0, 1, 2), "inputs": (0, 1, 2, 3, 4, 5)} for i in range(4)]

        for datatype_input, datatype_label in [
            (list, int),
            (list, float),
            (np.array, int),
            (np.array, np.array),
            (list, self._turn_to_none),
        ]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=lambda x: default_data_collator(x, return_tensors="np"),
                base_data=features_base_single_label,
                input_key="inputs",
                input_datatype=datatype_input,
                label_key="label",
                label_datatype=datatype_label,
            )

        for datatype_input, datatype_label in [(list, list), (list, self._turn_to_none)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=lambda x: default_data_collator(x, return_tensors="np"),
                base_data=features_base_multiple_labels,
                input_key="inputs",
                input_datatype=datatype_input,
                label_key="label",
                label_datatype=datatype_label,
            )

        features_base_single_label_alt = [{"input_ids": (0, 1, 2, 3, 4), "label": float(i)} for i in range(4)]
        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=lambda x: default_data_collator(x, return_tensors="np"),
            base_data=features_base_single_label_alt,
            input_key="input_ids",
            input_datatype=list,
            label_key="label",
            label_datatype=float,
        )

    def test_with_padding_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]
        features_batch = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10, return_tensors="np")
        self._validate_original_data_against_collated_data(
            collator=data_collator, original_data=features_original, batch_data=features_batch
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        self._validate_original_data_against_collated_data(
            collator=data_collator, original_data=features_original, batch_data=features_batch
        )

    def test_for_token_classification_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": (0, 1, 2), "labels": (0, 1, 2)},
            {"input_ids": (0, 1, 2, 3, 4, 5), "labels": (0, 1, 2, 3, 4, 5)},
        ]
        token_classification_collators = [
            DataCollatorForTokenClassification(tokenizer, return_tensors="np"),
            DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=10, return_tensors="np"),
            DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8, return_tensors="np"),
            DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1, return_tensors="np"),
        ]

        for datatype_input, datatype_label in [(list, list)]:
            for collator in token_classification_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                )

        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=token_classification_collators[-1],
            base_data=features_base,
            input_key="input_ids",
            input_datatype=datatype_input,
            label_key="labels",
            label_datatype=datatype_label,
            ignore_label=True,
        )

    def test_seq2seq_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(6)), "labels": list(range(6))},
        ]
        seq2seq_collators = [
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, return_tensors="np"),
            DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=7, return_tensors="np"),
            DataCollatorForSeq2Seq(
                tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8, return_tensors="np"
            ),
            DataCollatorForSeq2Seq(
                tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1, return_tensors="np"
            ),
        ]

        for datatype_input, datatype_label in [(list, list)]:
            for collator in seq2seq_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                )

        self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
            collator=seq2seq_collators[-1],
            base_data=features_base,
            input_key="input_ids",
            input_datatype=datatype_input,
            label_key="labels",
            label_datatype=datatype_label,
            ignore_label=True,
        )

        features_base_no_pad = [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(3)), "labels": list(range(3))},
        ]
        seq2seq_no_padding_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=PaddingStrategy.DO_NOT_PAD, return_tensors="np"
        )
        for datatype_input, datatype_label in [(list, list)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=seq2seq_no_padding_collator,
                base_data=features_base_no_pad,
                input_key="input_ids",
                input_datatype=datatype_input,
                label_key="labels",
                label_datatype=datatype_label,
            )

    def test_language_modelling_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base_no_pad = [
            {"input_ids": tuple(range(10)), "labels": (1,)},
            {"input_ids": tuple(range(10)), "labels": (1,)},
        ]
        features_base_pad = [
            {"input_ids": tuple(range(5)), "labels": (1,)},
            {"input_ids": tuple(range(5)), "labels": (1,)},
        ]
        lm_collators = [
            DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="np"),
            DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="np"),
            DataCollatorForLanguageModeling(tokenizer, return_tensors="np"),
            DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="np"),
        ]

        for datatype_input, datatype_label in [(list, list)]:
            for collator in lm_collators:
                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base_no_pad,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                    ignore_label=True,
                )

                self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                    collator=collator,
                    base_data=features_base_pad,
                    input_key="input_ids",
                    input_datatype=datatype_input,
                    label_key="labels",
                    label_datatype=datatype_label,
                    ignore_label=True,
                )

    def test_whole_world_masking_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_base = [
            {"input_ids": list(range(10)), "labels": (1,)},
            {"input_ids": list(range(10)), "labels": (1,)},
        ]
        whole_word_masking_collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="np")

        for datatype_input, datatype_label in [(list, list), (np.array, np.array)]:
            self._validate_original_data_against_collated_data_on_specified_keys_and_datatypes(
                collator=whole_word_masking_collator,
                base_data=features_base,
                input_key="input_ids",
                input_datatype=datatype_input,
                label_key="labels",
                label_datatype=datatype_label,
                ignore_label=True,
            )

    def test_permutation_language_modelling_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        plm_collator = DataCollatorForPermutationLanguageModeling(tokenizer, return_tensors="np")

        no_pad_features_original = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        no_pad_features_batch = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        self._validate_original_data_against_collated_data(
            collator=plm_collator, original_data=no_pad_features_original, batch_data=no_pad_features_batch
        )

        pad_features_original = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        pad_features_batch = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._validate_original_data_against_collated_data(
            collator=plm_collator, original_data=pad_features_original, batch_data=pad_features_batch
        )

    def test_next_sentence_prediction_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]
        features_batch = [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]

        nsp_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        self._validate_original_data_against_collated_data(
            collator=nsp_collator, original_data=features_original, batch_data=features_batch
        )

        nsp_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        self._validate_original_data_against_collated_data(
            collator=nsp_collator, original_data=features_original, batch_data=features_batch
        )

    def test_sentence_order_prediction_collator_immutability(self):
        tokenizer = BertTokenizer(self.vocab_file)

        features_original = [
            {
                "input_ids": np.array([0, 1, 2, 3, 4]),
                "token_type_ids": np.array([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]
        features_batch = [
            {
                "input_ids": np.array([0, 1, 2, 3, 4]),
                "token_type_ids": np.array([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]

        sop_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        self._validate_original_data_against_collated_data(
            collator=sop_collator, original_data=features_original, batch_data=features_batch
        )

        sop_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="np")
        self._validate_original_data_against_collated_data(
            collator=sop_collator, original_data=features_original, batch_data=features_batch
        )
