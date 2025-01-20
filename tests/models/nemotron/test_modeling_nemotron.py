# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Testing suite for the PyTorch Nemotron model."""

import tempfile
import unittest

import pytest

from transformers import NemotronConfig, is_torch_available
from transformers.testing_utils import (
    is_flaky,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        NemotronForCausalLM,
        NemotronForQuestionAnswering,
        NemotronForSequenceClassification,
        NemotronForTokenClassification,
        NemotronModel,
    )


class NemotronModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = NemotronConfig
        model_class = NemotronModel
        for_causal_lm_class = NemotronForCausalLM
        for_sequence_class = NemotronForSequenceClassification
        for_token_class = NemotronForTokenClassification


@require_torch
class NemotronModelTest(GemmaModelTest):
    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]
    all_model_classes = (
        (
            NemotronModel,
            NemotronForCausalLM,
            NemotronForSequenceClassification,
            NemotronForQuestionAnswering,
            NemotronForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (NemotronForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronModel,
            "text-classification": NemotronForSequenceClassification,
            "text-generation": NemotronForCausalLM,
            "zero-shot": NemotronForSequenceClassification,
            "question-answering": NemotronForQuestionAnswering,
            "token-classification": NemotronForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = NemotronForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = NemotronModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronConfig, hidden_size=37)

    @unittest.skip("Eager and SDPA do not produce the same outputs, thus this test fails")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @require_torch_sdpa
    @require_torch_accelerator
    @slow
    def test_sdpa_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(reason="Model does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa"
                )
                model_sdpa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_sdpa = model_sdpa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_sdpa = outputs_sdpa.hidden_states[-1]

                # nemotron sdpa needs a high tolerance
                assert torch.allclose(logits_sdpa, logits, atol=1e-2)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @is_flaky()
    @slow
    def test_flash_attn_2_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(reason="Model does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]

                # nemotron flash attention 2 needs a high tolerance
                assert torch.allclose(logits_fa, logits, atol=1e-2)


@require_torch_gpu
class NemotronIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    @require_read_token
    def test_nemotron_8b_generation_sdpa(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_nemotron_8b_generation_eager(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_nemotron_8b_generation_fa2(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)
