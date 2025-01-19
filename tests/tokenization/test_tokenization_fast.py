# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers

from ..test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class PreTrainedTokenizationFastTest(TokenizerTesterMixin, unittest.TestCase):
    rust_tokenizer_class = PreTrainedTokenizerFast
    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_vocab_key = "tokenizer_file"

    def setUp(self):
        self.test_rust_tokenizer = False  # because we don't have pretrained_vocab_files_map
        super().setUp()
        self.test_rust_tokenizer = True

        model_paths = ["robot-test/dummy-tokenizer-fast", "robot-test/dummy-tokenizer-wordlevel"]
        self.bytelevel_bpe_model_name = "SaulLu/dummy-tokenizer-bytelevel-bpe"

        # Inclusion of 2 tokenizers to test different types of models (Unigram and WordLevel for the moment)
        self.tokenizers_list = [(PreTrainedTokenizerFast, model_path, {}) for model_path in model_paths]

        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_paths[0])
        tokenizer.save_pretrained(self.tmpdirname)

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_tokenizer_mismatch_warning(self):
        pass

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_added_tokens_serialization(self):
        pass

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_additional_special_tokens_serialization(self):
        pass

    @unittest.skip(reason="PreTrainedTokenizerFast is the only tokenizer that is not linked to any model")
    def test_prepare_for_model(self):
        pass

    @unittest.skip(reason="PreTrainedTokenizerFast doesn't have tokenizer_file in its signature")
    def test_rust_tokenizer_signature(self):
        pass

    def test_training_new_tokenizer(self):
        tmpdirname_orig = self.tmpdirname
        # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                    tokenizer.save_pretrained(self.tmpdirname)
                    super().test_training_new_tokenizer()
                finally:
                    # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
                    # is restored
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_special_tokens_change(self):
        tmpdirname_orig = self.tmpdirname
        # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                    tokenizer.save_pretrained(self.tmpdirname)
                    super().test_training_new_tokenizer_with_special_tokens_change()
                finally:
                    # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
                    # is restored
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_bytelevel(self):
        tokenizer = self.rust_tokenizer_class.from_pretrained(self.bytelevel_bpe_model_name)

        toy_text_iterator = ("a" for _ in range(1000))
        new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=toy_text_iterator, length=1000, vocab_size=50)

        encoding_ids = new_tokenizer.encode("a🤗")
        self.assertEqual(encoding_ids, [64, 172, 253, 97, 245])

    def test_training_new_tokenizer_with_sequence_postprocessor(self):
        tokenizer_fast = self.rust_tokenizer_class.from_pretrained("hf-internal-testing/llama-3-8b-internal")

        toy_text_iterator = ("a" for _ in range(1000))
        new_tokenizer = tokenizer_fast.train_new_from_iterator(
            text_iterator=toy_text_iterator, length=1000, vocab_size=300
        )
        bos_token = "<|begin_of_text|>"
        new_bos_token_id = new_tokenizer._tokenizer.get_vocab(True)[bos_token]
        cfg = json.loads(new_tokenizer._tokenizer.to_str())
        # Check if "<|begin_of_text|>" id is changed correctly
        template_post_processor_json = cfg["post_processor"]["processors"][1]
        processor_token_id = template_post_processor_json["special_tokens"][bos_token]["ids"][0]
        self.assertEqual(processor_token_id, new_bos_token_id)
        # Check if "<|begin_of_text|>" is correct in encoded text
        self.assertEqual(new_tokenizer.encode("a🤗")[0], new_bos_token_id)

    def test_init_from_tokenizers_model(self):
        from tokenizers import Tokenizer

        sentences = ["Hello, y'all!", "How are you 😁 ? There should not be any issue right?"]

        tokenizer = Tokenizer.from_pretrained("google-t5/t5-base")
        # Enable padding
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>", length=512, pad_to_multiple_of=8)
        self.assertEqual(
            tokenizer.padding,
            {
                "length": 512,
                "pad_to_multiple_of": 8,
                "pad_id": 0,
                "pad_token": "<pad>",
                "pad_type_id": 0,
                "direction": "right",
            },
        )
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tmpdirname = tempfile.mkdtemp()
        fast_tokenizer.save_pretrained(tmpdirname)
        fast_from_saved = PreTrainedTokenizerFast.from_pretrained(tmpdirname)
        for tok in [fast_tokenizer, fast_from_saved]:
            self.assertEqual(tok.pad_token_id, 0)
            self.assertEqual(tok.padding_side, "right")
            self.assertEqual(tok.pad_token, "<pad>")
            self.assertEqual(tok.init_kwargs["max_length"], 512)
            self.assertEqual(tok.init_kwargs["pad_to_multiple_of"], 8)
            self.assertEqual(tok(sentences, padding = True), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1, 0, 0, 0, 0,0, 0, 0, 0],[ 571, 33, 25, 3, 2, 3, 58, 290, 225, 59, 36, 136, 962, 269, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip

        tokenizer.enable_truncation(8, stride=0, strategy="longest_first", direction="right")
        self.assertEqual(
            tokenizer.truncation, {"max_length": 8, "stride": 0, "strategy": "longest_first", "direction": "right"}
        )
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tmpdirname = tempfile.mkdtemp()
        fast_tokenizer.save_pretrained(tmpdirname)
        fast_from_saved = PreTrainedTokenizerFast.from_pretrained(tmpdirname)
        for tok in [fast_tokenizer, fast_from_saved]:
            self.assertEqual(tok.truncation_side, "right")
            self.assertEqual(tok.init_kwargs["truncation_strategy"], "longest_first")
            self.assertEqual(tok.init_kwargs["max_length"], 8)
            self.assertEqual(tok.init_kwargs["stride"], 0)
            # NOTE even if the model has a default max_length, it is not used...
            # thus tok(sentences, truncation = True) does nothing and does not warn either
            self.assertEqual(tok(sentences, truncation = True, max_length = 8), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1],[ 571, 33, 25, 3, 2, 3, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip


@require_tokenizers
class TokenizerVersioningTest(unittest.TestCase):
    def test_local_versioning(self):
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        json_tokenizer["model"]["vocab"]["huggingface"] = len(tokenizer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Hack to save this in the tokenizer_config.json
            tokenizer.init_kwargs["fast_tokenizer_files"] = ["tokenizer.4.0.0.json"]
            tokenizer.save_pretrained(tmp_dir)
            json.dump(json_tokenizer, open(os.path.join(tmp_dir, "tokenizer.4.0.0.json"), "w"))

            # This should pick the new tokenizer file as the version of Transformers is > 4.0.0
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer) + 1)
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertIn("huggingface", json_tokenizer["model"]["vocab"])

            # Will need to be adjusted if we reach v42 and this test is still here.
            # Should pick the old tokenizer file as the version of Transformers is < 4.0.0
            shutil.move(os.path.join(tmp_dir, "tokenizer.4.0.0.json"), os.path.join(tmp_dir, "tokenizer.42.0.0.json"))
            tokenizer.init_kwargs["fast_tokenizer_files"] = ["tokenizer.42.0.0.json"]
            tokenizer.save_pretrained(tmp_dir)
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer))
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])

    def test_repo_versioning(self):
        # This repo has two tokenizer files, one for v4.0.0 and above with an added token, one for versions lower.
        repo = "hf-internal-testing/test-two-tokenizers"

        # This should pick the new tokenizer file as the version of Transformers is > 4.0.0
        tokenizer = AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(tokenizer), 28997)
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        self.assertIn("huggingface", json_tokenizer["model"]["vocab"])

        # Testing an older version by monkey-patching the version in the module it's used.
        import transformers as old_transformers

        old_transformers.tokenization_utils_base.__version__ = "3.0.0"
        old_tokenizer = old_transformers.models.auto.AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(old_tokenizer), 28996)
        json_tokenizer = json.loads(old_tokenizer._tokenizer.to_str())
        self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])


@require_tokenizers
class ReduceMutableBorrowTests(unittest.TestCase):
    def test_async_share_tokenizer(self):
        # See https://github.com/huggingface/transformers/pull/12550
        # and https://github.com/huggingface/tokenizers/issues/537
        tokenizer = PreTrainedTokenizerFast.from_pretrained("robot-test/dummy-tokenizer-wordlevel")
        text = "The Matrix is a 1999 science fiction action film."

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fetch, tokenizer, text) for i in range(10)]
            return_value = [future.result() for future in futures]
            self.assertEqual(return_value, [[1, 10, 0, 8, 0, 18, 0, 0, 0, 2] for i in range(10)])

    def fetch(self, tokenizer, text):
        return tokenizer.encode(text, truncation="longest_first", padding="longest")
