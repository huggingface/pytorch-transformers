# coding=utf-8
# Copyright 2023 The HuggingFace Team Inc.
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

import tempfile
import unittest

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.testing_utils import (
    is_torch_available,
    require_optimum,
    require_torch,
    slow,
)


if is_torch_available():
    import torch


@require_torch
@require_optimum
@slow
class BetterTransformerIntegrationTest(unittest.TestCase):
    # refer to the full test suite in Optimum library:
    # https://github.com/huggingface/optimum/tree/main/tests/bettertransformer

    def test_transform_and_reverse(self):
        r"""
        Classic tests to simply check if the conversion has been successfull.
        """
        model_id = "hf-internal-testing/tiny-random-t5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        inp = tokenizer("This is me", return_tensors="pt")

        model = model.to_bettertransformer()

        self.assertTrue(any("BetterTransformer" in mod.__class__.__name__ for _, mod in model.named_modules()))

        output = model.generate(**inp)

        model = model.reverse_bettertransformer()

        self.assertFalse(any("BetterTransformer" in mod.__class__.__name__ for _, mod in model.named_modules()))

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            model_reloaded = AutoModelForSeq2SeqLM.from_pretrained(tmpdirname)

            self.assertFalse(
                any("BetterTransformer" in mod.__class__.__name__ for _, mod in model_reloaded.named_modules())
            )

            output_from_pretrained = model_reloaded.generate(**inp)
            torch.testing.assert_close(output, output_from_pretrained)

    def test_error_save_pretrained(self):
        r"""
        The save_pretrained method should raise a ValueError if the model is in BetterTransformer mode.
        All should be good if the model is reversed.
        """
        model_id = "hf-internal-testing/tiny-random-t5"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        model = model.to_bettertransformer()

        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError):
                model.save_pretrained(tmpdirname)

            model = model.reverse_bettertransformer()
            model.save_pretrained(tmpdirname)
