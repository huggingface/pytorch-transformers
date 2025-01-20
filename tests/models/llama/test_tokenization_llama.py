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

import os
import pickle
import shutil
import tempfile
import unittest

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import (
    SPIECE_UNDERLINE,
    AddedToken,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_jinja,
    require_read_token,
    require_sentencepiece,
    require_tiktoken,
    require_tokenizers,
    require_torch,
    slow,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class LlamaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["hf-internal-testing/llama-tokenizer", "meta-llama/Llama-2-7b-hf"]
    tokenizer_class = LlamaTokenizer
    rust_tokenizer_class = LlamaTokenizerFast

    test_rust_tokenizer = False
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = LlamaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizers(self, **kwargs):
        kwargs.update({"pad_token": "<PAD>"})
        return super().get_tokenizers(**kwargs)

    def test_full_tokenizer(self):
        tokenizer = LlamaTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    @unittest.skip(reason="Let's wait for the fast tokenizer!")
    def test_save_pretrained(self):
        self.tokenizers_list += (self.rust_tokenizer_class, "hf-internal-testing/llama-tokenizer", {})
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files + the tokenizer.json file for the fast one
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))
                tokenizer_r_files = tuple(f for f in tokenizer_r_files if "tokenizer.json" not in f)
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=True
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=True)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=False
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=False)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it saved the tokenizer.json file
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

    @require_torch
    def test_batch_tokenization(self):
        if not self.test_seq2seq:
            self.skipTest(reason="test_seq2seq is set to False")

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Longer text that will definitely require truncation.
                text = [
                    " UN Chief Says There Is No Military Solution in Syria",
                    " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for"
                    " Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons"
                    " will only worsen the violence and misery for millions of people.",
                ]
                try:
                    batch = tokenizer(
                        text=text,
                        max_length=3,
                        max_target_length=10,
                        return_tensors="pt",
                    )
                except NotImplementedError:
                    self.skipTest(reason="Encountered NotImplementedError when calling tokenizer")
                self.assertEqual(batch.input_ids.shape[1], 3)
                # max_target_length will default to max_length if not specified
                batch = tokenizer(text, max_length=3, return_tensors="pt")
                self.assertEqual(batch.input_ids.shape[1], 3)

                batch_encoder_only = tokenizer(text=text, max_length=3, max_target_length=10, return_tensors="pt")
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    @unittest.skip(reason="Unfortunately way too slow to build a BPE with SentencePiece.")
    def test_save_slow_from_fast_and_reload_fast(self):
        pass

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name,
                        additional_special_tokens=added_tokens,
                        **kwargs,  # , from_slow=True <- unfortunately too slow to convert
                    )
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    p_output = tokenizer_p.encode("Hey this is a <special> token")

                    cr_output = tokenizer_cr.encode("Hey this is a <special> token")

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    @slow
    def test_tokenizer_integration(self):
        expected_encoding = {'input_ids': [[1, 4103, 689, 414, 313, 24784, 368, 2998, 408, 282, 3637, 25350, 29899, 9067, 414, 322, 282, 3637, 25350, 29899, 1457, 3018, 1312, 29899, 2151, 29897, 8128, 2498, 29899, 15503, 4220, 6956, 1973, 313, 13635, 29911, 29892, 402, 7982, 29899, 29906, 29892, 1528, 13635, 29911, 29874, 29892, 1060, 26369, 29892, 6652, 309, 29933, 814, 29892, 1060, 29931, 6779, 11410, 363, 18385, 17088, 7634, 11235, 313, 25103, 29965, 29897, 322, 18385, 17088, 28203, 313, 25103, 29954, 29897, 411, 975, 29871, 29941, 29906, 29974, 758, 3018, 1312, 4733, 297, 29871, 29896, 29900, 29900, 29974, 10276, 322, 6483, 1006, 3372, 3097, 1546, 435, 1165, 29892, 10772, 29911, 25350, 322, 323, 6073, 17907, 29889], [1, 350, 20161, 338, 8688, 304, 758, 29899, 14968, 6483, 21000, 8684, 284, 22540, 515, 443, 29880, 24025, 1426, 491, 14002, 368, 4195, 292, 373, 1716, 2175, 322, 1492, 3030, 297, 599, 15359, 29889], [1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="hf-internal-testing/llama-tokenizer",
            revision="0984d03108b1a041ed679bd253b6519b7e1a4778",
            padding=False,
        )

    def test_picklable(self):
        with tempfile.NamedTemporaryFile() as f:
            shutil.copyfile(SAMPLE_VOCAB, f.name)
            tokenizer = LlamaTokenizer(f.name, keep_accents=True)
            pickled_tokenizer = pickle.dumps(tokenizer)
        pickle.loads(pickled_tokenizer)

    @unittest.skip(reason="worker 'gw4' crashed on CI, passing locally.")
    def test_pickle_subword_regularization_tokenizer(self):
        pass

    @unittest.skip(reason="worker 'gw4' crashed on CI, passing locally.")
    def test_subword_regularization_tokenizer(self):
        pass

    def test_add_prefix_space(self):
        pretrained_name = "hf-internal-testing/llama-tokenizer-non-normalized"
        inputs = "Hey how are you doing"
        EXPECTED_WITH_SPACE = [1, 18637, 920, 526, 366, 2599]
        EXPECTED_WO_SPACE = [1, 29950, 1032, 920, 526, 366, 2599]

        slow_ = self.tokenizer_class.from_pretrained(pretrained_name, add_prefix_space=False, legacy=False)
        fast_ = self.rust_tokenizer_class.from_pretrained(pretrained_name, add_prefix_space=False, legacy=False)
        self.assertEqual(slow_.encode(inputs), EXPECTED_WO_SPACE)
        self.assertEqual(slow_.encode(inputs), fast_.encode(inputs))
        self.assertEqual(slow_.tokenize(inputs), ["H", "ey", "▁how", "▁are", "▁you", "▁doing"])
        self.assertEqual(slow_.decode(EXPECTED_WO_SPACE, skip_special_tokens=True), inputs)
        self.assertEqual(
            slow_.decode(EXPECTED_WO_SPACE, skip_special_tokens=True),
            fast_.decode(EXPECTED_WO_SPACE, skip_special_tokens=True),
        )

        slow_ = self.tokenizer_class.from_pretrained(pretrained_name, add_prefix_space=True, legacy=False)
        fast_ = self.rust_tokenizer_class.from_pretrained(pretrained_name, add_prefix_space=True, legacy=False)
        self.assertEqual(slow_.encode(inputs), EXPECTED_WITH_SPACE)
        self.assertEqual(slow_.encode(inputs), fast_.encode(inputs))
        self.assertEqual(slow_.tokenize(inputs), ["▁Hey", "▁how", "▁are", "▁you", "▁doing"])
        self.assertEqual(slow_.decode(EXPECTED_WITH_SPACE, skip_special_tokens=True), inputs)
        self.assertEqual(
            slow_.decode(EXPECTED_WITH_SPACE, skip_special_tokens=True),
            fast_.decode(EXPECTED_WITH_SPACE, skip_special_tokens=True),
        )

    def test_load_tokenizer_with_model_file_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_hub_download(repo_id="huggyllama/llama-7b", filename="tokenizer.model", local_dir=tmp_dir)
            tokenizer_fast = self.rust_tokenizer_class.from_pretrained(tmp_dir)
            self.assertEqual(tokenizer_fast.encode("This is a test"), [1, 910, 338, 263, 1243])

            tokenizer_slow = self.tokenizer_class.from_pretrained(tmp_dir)
            self.assertEqual(tokenizer_slow.encode("This is a test"), [1, 910, 338, 263, 1243])


@require_torch
@require_sentencepiece
@require_tokenizers
class LlamaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "hf-internal-testing/llama-tokenizer-non-normalized"
        cls.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(checkpoint_name)
        cls.rust_tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint_name)
        return cls

    @require_torch
    def integration_tests(self):
        inputs = self.tokenizer(
            ["The following string should be properly encoded: Hello.", "But ird and ปี   ird   ด"],
            return_tensors="pt",
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                "input_ids": [
                    [1, 450, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889],
                    [1, 1205, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718],
                ],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            },
        )

    def test_fast_special_tokens(self):
        slow_tokenizer = self.tokenizer
        fast_tokenizer = self.rust_tokenizer
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [1, 319, 4559, 1243]

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 319, 4559, 1243]

        fast_tokenizer.add_eos_token = True
        print(fast_tokenizer.add_eos_token)
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 319, 4559, 1243, 2]

        slow_tokenizer.add_eos_token = True
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [1, 319, 4559, 1243, 2]

        fast_tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer", add_eos_token=True, add_bos_token=False
        )
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [319, 4559, 1243, 2]

        slow_tokenizer = LlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", add_eos_token=True, add_bos_token=False
        )
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [319, 4559, 1243, 2]

        self.tokenizer.add_eos_token = False
        self.rust_tokenizer.add_eos_token = False

    @slow
    def test_conversion(self):
        # This is excruciatingly slow since it has to recreate the entire merge
        # list from the original vocabulary in spm
        self.rust_tokenizer.save_pretrained("./out")
        with tempfile.TemporaryDirectory() as dirname:
            self.rust_tokenizer.save_pretrained(dirname)

            with open(os.path.join(dirname, "tokenizer.json"), "r") as f:
                old_serialized = f.read()

        new_tokenizer = convert_slow_tokenizer(self.tokenizer)
        with tempfile.NamedTemporaryFile() as f:
            new_tokenizer.save(f.name)
            # Re-opening since `f` is in bytes.
            new_serialized = open(f.name, "r").read()
            with open("out_tokenizer.json", "w") as g:
                g.write(new_serialized)

            self.assertEqual(old_serialized, new_serialized)

    def test_simple_encode_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(rust_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(pyth_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")
        self.assertEqual(rust_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(pyth_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])  # fmt: skip
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])  # fmt: skip
        self.assertEqual(
            pyth_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )
        self.assertEqual(
            rust_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(pyth_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(pyth_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")
        self.assertEqual(rust_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(pyth_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(pyth_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")
        self.assertEqual(rust_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode(" "), [1, 259])
        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(pyth_tokenizer.encode("  "), [1, 1678])
        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [1, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

    def test_no_differences_showcase(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode(" "), [1, 259])
        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(pyth_tokenizer.encode("  "), [1, 1678])
        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [1, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    def test_no_differences_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.decode([869]), ".")
        self.assertEqual(rust_tokenizer.decode([869]), ".")

        self.assertEqual(pyth_tokenizer.decode([30112, 869]), "ا .")
        self.assertEqual(rust_tokenizer.decode([30112, 869]), "ا .")

    def test_no_differences_special_tokens(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    @unittest.skipIf(
        os.getenv("RUN_TOKENIZER_INTEGRATION", "0") == "0",
        "RUN_TOKENIZER_INTEGRATION=1 to run tokenizer integration tests",
    )
    def test_integration_test_xnli(self):
        import tqdm

        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        dataset = load_dataset("google/code_x_glue_ct_code_to_text", "go")
        for item in tqdm.tqdm(dataset["validation"]):
            string = item["code"]
            encoded1 = pyth_tokenizer.encode(string)
            encoded2 = rust_tokenizer.encode(string)

            self.assertEqual(encoded1, encoded2)

            decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("facebook/xnli", "all_languages")

        for item in tqdm.tqdm(dataset["train"]):
            for string in item["premise"].values():
                encoded1 = pyth_tokenizer.encode(string)
                encoded2 = rust_tokenizer.encode(string)

                self.assertEqual(encoded1, encoded2)

                decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
                decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

                self.assertEqual(decoded1, decoded2)

    def test_special_token_special_word(self):
        # the word inform should be split as ['in', 'form']
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)

        example_inputs = tokenizer.tokenize("<REPR_END>inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["<REPR_END>", "in", "form", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        # Make sure dummy space is added if it is indeed the first word
        example_inputs = tokenizer.tokenize("inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["▁inform", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])
        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>inform")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=True
        )
        # decoding strips the added prefix space.
        self.assertEqual(out2, "<REPR_END>inform")
        input_ids = tokenizer.encode("<REPR_END>inform", add_special_tokens=False)
        self.assertEqual(input_ids, [32000, 262, 689])  # 29871 is the spiece underline, '▁' added as it should

        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        # TODO @ArthurZ currently we strip left and right, so this will not keep the spaces
        self.assertEqual(out2, "<REPR_END>inform")

        ### Let's make sure decoding does not add extra spaces here and there
        # TODO @ArthurZ this should be affected by the lstrip/rstrip/single word /normalize refactoring
        # Since currently we always strip left and right of the token, results are as such
        input_ids = tokenizer.encode("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(input_ids, [1, 15043, 1, 3525])
        tokens = tokenizer.tokenize("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(tokens, ["<s>", "▁Hello", "<s>", "how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s>how")

        # Let's make sure that if there are any spaces, we don't remove them!
        input_ids = tokenizer.encode(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(input_ids, [29871, 1, 15043, 1, 920])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s> how")

        # Let's make sure the space is preserved
        input_ids = tokenizer.encode("hello", add_special_tokens=True)
        self.assertEqual(input_ids, [1, 22172])
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, ["▁hello"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> hello")

        input_ids = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(input_ids, [22172])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "hello")

    def test_no_prefix_space(self):
        tokenizer_no_prefix_space = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", add_prefix_space=False)
        no_prefix_space_tokens = tokenizer_no_prefix_space.tokenize("Hey")
        self.assertEqual(no_prefix_space_tokens, ["H", "ey"])

        tokenizer = LlamaTokenizerFast.from_pretrained(
            "huggyllama/llama-7b", legacy=False, from_slow=True, add_prefix_space=False
        )
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)

        example_inputs = tokenizer.tokenize("<REPR_END>inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["<REPR_END>", "in", "form", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        # Make sure dummy space is added if it is indeed the first word
        example_inputs = tokenizer.tokenize("inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["in", "form", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])
        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>inform")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=True
        )
        # decoding strips the added prefix space.
        self.assertEqual(out2, "<REPR_END>inform")
        input_ids = tokenizer.encode("<REPR_END>inform", add_special_tokens=False)
        self.assertEqual(input_ids, [32000, 262, 689])  # 29871 is the spiece underline, '▁' added as it should

        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out2, "<REPR_END>inform")

        input_ids = tokenizer.encode("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(input_ids, [1, 15043, 1, 3525])
        tokens = tokenizer.tokenize("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(tokens, ["<s>", "▁Hello", "<s>", "how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s>how")

        # Let's make sure that if there are any spaces, we don't remove them!
        input_ids = tokenizer.encode(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(input_ids, [29871, 1, 15043, 1, 920])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, " <s> Hello<s> how")

        # Let's make sure the space is preserved
        input_ids = tokenizer.encode("hello", add_special_tokens=True)
        self.assertEqual(input_ids, [1, 12199])
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, ["hello"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s>hello")

        input_ids = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(input_ids, [12199])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "hello")

    def test_some_edge_cases(self):
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)

        sp_tokens = tokenizer.sp_model.encode("<s>>", out_type=str)
        self.assertEqual(sp_tokens, ["<", "s", ">>"])
        tokens = tokenizer.tokenize("<s>>")
        self.assertNotEqual(sp_tokens, tokens)
        self.assertEqual(tokens, ["<s>", ">"])

        tokens = tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode("", out_type=str))

        tokens = tokenizer.tokenize(" ")
        self.assertEqual(tokens, ["▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, tokenizer.sp_model.encode("  ", out_type=str))

        tokens = tokenizer.tokenize("▁")
        self.assertEqual(tokens, ["▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁▁", out_type=str))

        tokens = tokenizer.tokenize(" ▁")
        self.assertEqual(tokens, ["▁▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁▁▁", out_type=str))

    def test_fast_post_processor(self):
        tokenizer = LlamaTokenizerFast(
            SAMPLE_VOCAB, eos_token=None, bos_token=None, add_bos_token=False, add_eos_token=False
        )
        tokenizer.encode(" Hey ")

        with self.assertRaises(ValueError):
            tokenizer = LlamaTokenizerFast(
                SAMPLE_VOCAB, bos_token=None, eos_token="<s>", add_bos_token=True, add_eos_token=False
            )
        with self.assertRaises(ValueError):
            tokenizer = LlamaTokenizerFast(SAMPLE_VOCAB, eos_token=None, add_bos_token=True, add_eos_token=True)

    @require_jinja
    def test_tokenization_for_chat(self):
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)

        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
            [{"role": "user", "content": "Hello!"}],
        ]
        # Matt: The third test case tests the default system message, but if this is ever changed in the
        #       class/repo code then that test will fail, and the case will need to be updated.
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        # fmt: off
        expected_tokens = [
            [1, 29961, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 263, 8444, 13563, 7451, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 10994, 29991, 518, 29914, 25580, 29962],
            [1, 29961, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 263, 8444, 13563, 7451, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 10994, 29991, 518, 29914, 25580, 29962, 20103, 304, 5870, 366, 29889, 29871, 2],
            [1, 29961, 25580, 29962, 15043, 29991, 518, 29914, 25580, 29962]
        ]
        # fmt: on
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)


@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    @classmethod
    def setUpClass(cls):
        tokenizer = LlamaTokenizer(SAMPLE_VOCAB, extra_ids=0, add_bos_token=False, legacy=False)
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("<s>", rstrip=False, lstrip=False)]})
        cls.tokenizer = tokenizer
        return cls

    def test_add_dummy_prefix(self):
        # make sure `'▁'` is prepended, and outputs match sp_model's
        # `sentencepiece.NormalizerSpec.add_dummy_prefix` attribute
        input_ids = self.tokenizer.encode(". Hello")
        self.assertEqual(input_ids, [7, 4, 156, 86, 20])
        sp_encode = self.tokenizer.sp_model.encode(". Hello")
        self.assertEqual(input_ids, [7] + sp_encode)
        tokens = self.tokenizer.tokenize(". Hello")
        self.assertEqual(tokens, ["▁", ".", "▁He", "ll", "o"])

        tokens = self.tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode("", out_type=str))

        tokens = self.tokenizer.tokenize(" ")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode(" ", out_type=str))

        tokens = self.tokenizer.tokenize("▁")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode("▁", out_type=str))

    def test_remove_extra_whitespaces(self):
        # make sure the extra spaces are eaten. Since the sample vocab does not have
        # `______`. sentencepiece.NormalizerSpec.remove_extra_whitespaces attribute is set to False

        input_ids = self.tokenizer.encode("       . Hello")
        self.assertEqual(input_ids, [7, 4, 156, 86, 20])
        sp_encode = self.tokenizer.sp_model.encode("       . Hello")
        self.assertEqual(input_ids, [7] + sp_encode)
        tokens = self.tokenizer.tokenize(" . Hello")
        self.assertEqual(tokens, ["▁", ".", "▁He", "ll", "o"])

        # `'▁'` is also a whitespace
        input_ids = self.tokenizer.encode("▁He is not")
        self.assertEqual(input_ids, [156, 46, 44])
        tokens = self.tokenizer.tokenize("▁He is not")
        sp_encode = [
            self.tokenizer.sp_model.piece_to_id("▁He"),
            self.tokenizer.sp_model.piece_to_id("▁is"),
            self.tokenizer.sp_model.piece_to_id("▁not"),
        ]
        self.assertEqual(input_ids, sp_encode)
        self.assertEqual(tokens, ["▁He", "▁is", "▁not"])  # no extra space added

        input_ids = self.tokenizer.encode("▁He is not<s>             ▁He")
        self.assertEqual(input_ids, [156, 46, 44, 1, 156])
        tokens = self.tokenizer.tokenize("▁He is not<s>              ▁He")
        self.assertEqual(tokens, ["▁He", "▁is", "▁not", "<s>", "▁He"])  # spaces are eaten by spm + our strip
        # make sure that the output after the extra id is the same as if
        # extra_id was not there
        input_ids = self.tokenizer.encode("▁He is not             ▁He")
        self.assertEqual(input_ids, [156, 46, 44, 156])
        tokens = self.tokenizer.tokenize("▁He is not              ▁He")
        self.assertEqual(tokens, ["▁He", "▁is", "▁not", "▁He"])  # spaces are eaten by spm even if not start

    def test_character_after_special_token(self):
        # Make sure that `tokenizer.tokenize` is similar to
        # adding the equivalent special token to the vocab
        input_ids = self.tokenizer.encode("Hey <s>I")
        self.assertEqual(input_ids, [156, 30, 1, 100])
        sp_encode = self.tokenizer.sp_model.encode("Hey .I")
        # the last token should be 100
        self.assertEqual(input_ids[-1], sp_encode[-1])
        tokens = self.tokenizer.tokenize("<s>I")
        self.assertEqual(tokens, ["<s>", "I"])

        input_ids = self.tokenizer.encode("Hello, <s>,")
        self.assertEqual(input_ids, [156, 86, 20, 3, 1, 3])
        tokens = self.tokenizer.tokenize("Hello, <s>,")
        self.assertEqual(tokens, ["▁He", "ll", "o", ",", "<s>", ","])

    def test_special_tokens_strip(self):
        input_ids = self.tokenizer.encode(" <s> ,")
        self.assertEqual(input_ids, [1, 7, 3])
        tokens = self.tokenizer.tokenize(" <s> ,")
        # spaces are eaten by rstrip / lstrip + spm sp_model.encode("  ") = []
        self.assertEqual(tokens, ["<s>", "▁", ","])

        input_ids = self.tokenizer.encode("No <s> ▁He")
        self.assertEqual(input_ids, [284, 1, 156])
        tokens = self.tokenizer.tokenize("No <s> ▁He")
        self.assertEqual(tokens, ["▁No", "<s>", "▁He"])  # spaces are eaten by rstrip / lstrip

    @require_read_token
    def test_bos_eos_tokens(self):
        new_eos_token = "<new_eos>"
        model_path = "hf-internal-testing/llama-3-8b-internal"
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False, add_eos_token=True)
        self.assertNotEqual(tokenizer("hello")["input_ids"][0], tokenizer.bos_token_id)  # no bos token
        self.assertEqual(tokenizer("hello")["input_ids"][-1], tokenizer.eos_token_id)  # eos token

        tokenizer.add_special_tokens({"eos_token": new_eos_token})  # update new eos token
        tokens = tokenizer.tokenize("hello", add_special_tokens=True)
        self.assertEqual(tokens[-1], new_eos_token)

        self.assertEqual(tokenizer("hello")["input_ids"][0], tokenizer.bos_token_id)
        self.assertEqual(tokenizer("hello")["input_ids"][-1], tokenizer.eos_token_id)

        tokenizer.add_special_tokens({"eos_token": new_eos_token})  # update new eos token
        tokens = tokenizer.tokenize("hello", add_special_tokens=True)
        self.assertEqual(tokens[-1], new_eos_token)

        tmpdirname = tempfile.mkdtemp()
        tokenizer.save_pretrained(tmpdirname)
        tokenizer_reload = AutoTokenizer.from_pretrained(tmpdirname)

        self.assertTrue(isinstance(tokenizer_reload, PreTrainedTokenizerFast))
        tokens = tokenizer_reload.tokenize("hello", add_special_tokens=True)
        self.assertEqual(tokens[-1], new_eos_token)
        shutil.rmtree(tmpdirname)


@require_tiktoken
@require_read_token
class TikTokenIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    def test_tiktoken_llama(self):
        model_path = "hf-internal-testing/llama-3-8b-internal"
        subfolder = "original"
        test_text = "This is a test sentence."
        test_tokens = [128000, 2028, 374, 264, 1296, 11914, 13, 128001]
        num_reserved_special_tokens = 256
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
            "<|python_tag|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, num_reserved_special_tokens - 5)]

        tiktoken_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_path,
            subfolder=subfolder,
            additional_special_tokens=special_tokens,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
        )
        tokens = tiktoken_tokenizer.tokenize("<|begin_of_text|> " + test_text)
        self.assertEqual(tokens[0], "<|begin_of_text|>")

        tiktoken_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder=subfolder,
            legacy=False,
            additional_special_tokens=special_tokens,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            add_bos_token=True,
            add_eos_token=True,
        )
        self.assertTrue(isinstance(tiktoken_tokenizer, PreTrainedTokenizerFast))

        tokens = tiktoken_tokenizer.encode(test_text, add_special_tokens=True)
        self.assertEqual(tokens, test_tokens)

        tmpdirname = tempfile.mkdtemp()
        tiktoken_tokenizer.save_pretrained(tmpdirname)
        tokenizer_reload = AutoTokenizer.from_pretrained(tmpdirname)

        self.assertTrue(isinstance(tokenizer_reload, PreTrainedTokenizerFast))
        tokens = tokenizer_reload.encode(test_text, add_special_tokens=True)
        self.assertEqual(tokens, test_tokens)
        shutil.rmtree(tmpdirname)

        tiktoken_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder=subfolder,
            additional_special_tokens=special_tokens,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            from_slow=True,
            add_bos_token=True,
            add_eos_token=True,
        )
        tokens = tiktoken_tokenizer.encode(test_text, add_special_tokens=True)
        self.assertEqual(tokens, test_tokens)
