# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch CPMAnt model."""

import unittest

from transformers.testing_utils import is_torch_available, require_torch, tooslow

from ...generation.test_utils import torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        CpmAntConfig,
        CpmAntForCausalLM,
        CpmAntModel,
        CpmAntTokenizer,
    )


@require_torch
class CpmAntModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        is_training=True,
        use_token_type_ids=False,
        use_input_mask=False,
        use_labels=False,
        use_mc_token_ids=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        num_buckets=32,
        max_distance=128,
        prompt_length=8,
        prompt_types=8,
        segment_types=8,
        init_std=0.02,
        return_dict=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.prompt_length = prompt_length
        self.prompt_types = prompt_types
        self.segment_types = segment_types
        self.init_std = init_std
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = {}
        input_ids["input_ids"] = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).type(torch.int32)
        input_ids["use_cache"] = False

        config = self.get_config()

        return (config, input_ids)

    def get_config(self):
        return CpmAntConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            dim_ff=self.intermediate_size,
            position_bias_num_buckets=self.num_buckets,
            position_bias_max_distance=self.max_distance,
            prompt_types=self.prompt_types,
            prompt_length=self.prompt_length,
            segment_types=self.segment_types,
            use_cache=True,
            init_std=self.init_std,
            return_dict=self.return_dict,
        )

    def create_and_check_cpmant_model(self, config, input_ids, *args):
        model = CpmAntModel(config=config)
        model.to(torch_device)
        model.eval()

        hidden_states = model(**input_ids).last_hidden_state

        self.parent.assertEqual(hidden_states.shape, (self.batch_size, self.seq_length, config.hidden_size))

    def create_and_check_lm_head_model(self, config, input_ids, *args):
        model = CpmAntForCausalLM(config)
        model.to(torch_device)
        input_ids["input_ids"] = input_ids["input_ids"].to(torch_device)
        model.eval()

        model_output = model(**input_ids)
        self.parent.assertEqual(
            model_output.logits.shape,
            (self.batch_size, self.seq_length, config.vocab_size + config.prompt_types * config.prompt_length),
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class CpmAntModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CpmAntModel, CpmAntForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": CpmAntModel, "text-generation": CpmAntForCausalLM} if is_torch_available() else {}
    )

    test_pruning = False
    test_missing_keys = False
    test_mismatched_shapes = False
    test_head_masking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = CpmAntModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CpmAntConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        unittest.skip(reason="CPMAnt doesn't support input_embeds.")(self.test_inputs_embeds)

    def test_retain_grad_hidden_states_attentions(self):
        unittest.skip(
            "CPMAnt doesn't support retain grad in hidden_states or attentions, because prompt management will peel off the output.hidden_states from graph.\
                 So is attentions. We strongly recommand you use loss to tune model."
        )(self.test_retain_grad_hidden_states_attentions)

    def test_cpmant_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cpmant_model(config, inputs)

    def test_cpmant_lm_head_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(config, inputs)


@require_torch
class CpmAntModelIntegrationTest(unittest.TestCase):
    @tooslow
    def test_inference_masked_lm(self):
        texts = "今天天气真好！"
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntModel.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).last_hidden_state

        expected_slice = torch.tensor(
            [[[6.1708, 5.9244, 1.0835], [6.5207, 6.2893, -11.3324], [-1.0107, -0.0576, -5.9577]]],
        )
        torch.testing.assert_close(hidden_states[:, :3, :3], expected_slice, rtol=1e-2, atol=1e-2)


@require_torch
class CpmAntForCausalLMlIntegrationTest(unittest.TestCase):
    @tooslow
    def test_inference_casual(self):
        texts = "今天天气真好！"
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntForCausalLM.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).logits

        expected_slice = torch.tensor(
            [[[-6.4267, -6.4083, -6.3958], [-5.8802, -5.9447, -5.7811], [-5.3896, -5.4820, -5.4295]]],
        )
        torch.testing.assert_close(hidden_states[:, :3, :3], expected_slice, rtol=1e-2, atol=1e-2)

    @tooslow
    def test_simple_generation(self):
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntForCausalLM.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        texts = "今天天气不错，"
        expected_output = "今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的"
        model_inputs = tokenizer(texts, return_tensors="pt")
        token_ids = model.generate(**model_inputs)
        output_texts = tokenizer.batch_decode(token_ids)
        self.assertEqual(expected_output, output_texts)

    @tooslow
    def test_batch_generation(self):
        model_path = "openbmb/cpm-ant-10b"
        model = CpmAntForCausalLM.from_pretrained(model_path)
        tokenizer = CpmAntTokenizer.from_pretrained(model_path)
        texts = ["今天天气不错，", "新年快乐，万事如意！"]
        expected_output = [
            "今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的",
            "新年快乐，万事如意！在这辞旧迎新的美好时刻，我谨代表《农村新技术》杂志社全体同仁，向一直以来关心、支持《农村新技术》杂志发展的各级领导、各界朋友和广大读者致以最诚挚的",
        ]
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
        token_ids = model.generate(**model_inputs)
        output_texts = tokenizer.batch_decode(token_ids)
        self.assertEqual(expected_output, output_texts)
