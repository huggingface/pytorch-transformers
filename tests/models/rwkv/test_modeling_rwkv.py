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


import unittest
from unittest.util import safe_repr

from transformers import AutoTokenizer, RwkvConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        RwkvForCausalLM,
        RwkvModel,
    )


class RwkvModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_token_type_ids=False,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
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
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def get_large_model_config(self):
        return RwkvConfig.from_pretrained("sgugger/rwkv-4-pile-7b")

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config(
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

        return (
            config,
            input_ids,
            input_mask,
            None,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        return RwkvConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            activation_function=self.hidden_act,
            resid_pdrop=self.hidden_dropout_prob,
            attn_pdrop=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_rwkv_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        config.output_hidden_states = True
        model = RwkvModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def create_and_check_causl_lm(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = RwkvForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_state_equivalency(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = RwkvModel(config=config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids)
        output_whole = outputs.last_hidden_state

        outputs = model(input_ids[:, :2])
        output_one = outputs.last_hidden_state

        # Using the state computed on the first inputs, we will get the same output
        outputs = model(input_ids[:, 2:], state=outputs.state)
        output_two = outputs.last_hidden_state

        self.parent.assertTrue(torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5))

    def create_and_check_forward_and_backwards(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args, gradient_checkpointing=False
    ):
        model = RwkvForCausalLM(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids}

        return config, inputs_dict


@require_torch
class RwkvModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (RwkvModel, RwkvForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": RwkvModel, "text-generation": RwkvForCausalLM} if is_torch_available() else {}
    )
    all_generative_model_classes = (RwkvForCausalLM,) if is_torch_available() else ()
    fx_compatible = False
    test_missing_keys = False
    test_model_parallel = False
    test_pruning = False
    test_head_masking = False  # Rwkv does not support head masking

    def setUp(self):
        self.model_tester = RwkvModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=RwkvConfig, n_embd=37, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def assertInterval(self, member, container, msg=None):
        r"""
        Simple utility function to check if a member is inside an interval.
        """
        if isinstance(member, torch.Tensor):
            max_value, min_value = member.max().item(), member.min().item()
        elif isinstance(member, list) or isinstance(member, tuple):
            max_value, min_value = max(member), min(member)

        if not isinstance(container, list):
            raise TypeError("container should be a list or tuple")
        elif len(container) != 2:
            raise ValueError("container should have 2 elements")

        expected_min, expected_max = container

        is_inside_interval = (min_value >= expected_min) and (max_value <= expected_max)

        if not is_inside_interval:
            standardMsg = "%s not found in %s" % (safe_repr(member), safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_rwkv_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_rwkv_model(*config_and_inputs)

    def test_rwkv_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causl_lm(*config_and_inputs)

    def test_state_equivalency(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_state_equivalency(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                if "time_decay" in name:
                    if param.requires_grad:
                        self.assertTrue(param.data.max().item() == 3.0)
                        self.assertTrue(param.data.min().item() == -5.0)
                elif "time_first" in name:
                    if param.requires_grad:
                        # check if it's a ones like
                        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data), atol=1e-5, rtol=1e-5))
                elif any(x in name for x in ["time_mix_key", "time_mix_receptance"]):
                    if param.requires_grad:
                        self.assertInterval(
                            param.data,
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                elif "time_mix_value" in name:
                    if param.requires_grad:
                        self.assertInterval(
                            param.data,
                            [0.0, 1.3],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the attention outputs of Rwkv are different from other models
        it has a shape `batch_size, seq_len, hidden_size`.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            batch_size = inputs["input_ids"].shape[0]
            with torch.no_grad():
                outputs = model(**inputs)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            batch_size = inputs["input_ids"].shape[0]
            with torch.no_grad():
                outputs = model(**inputs)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [batch_size, seq_len, config.hidden_size],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            batch_size = inputs["input_ids"].shape[0]
            with torch.no_grad():
                outputs = model(**inputs)

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [batch_size, seq_len, config.hidden_size],
            )

    @slow
    def test_model_from_pretrained(self):
        model_name = "RWKV/rwkv-4-169m-pile"
        model = RwkvModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_beam_sample_generate_dict_output(self):
        # This model has a custom attention output shape AND config flags, let's skip those checks
        old_has_attentions = self.has_attentions
        self.has_attentions = False
        super().test_beam_sample_generate_dict_output()
        self.has_attentions = old_has_attentions

    def test_beam_search_generate_dict_output(self):
        # This model has a custom attention output shape AND config flags, let's skip those checks
        old_has_attentions = self.has_attentions
        self.has_attentions = False
        super().test_beam_search_generate_dict_output()
        self.has_attentions = old_has_attentions

    def test_constrained_beam_search_generate_dict_output(self):
        # This model has a custom attention output shape AND config flags, let's skip those checks
        old_has_attentions = self.has_attentions
        self.has_attentions = False
        super().test_constrained_beam_search_generate_dict_output()
        self.has_attentions = old_has_attentions

    def test_greedy_generate_dict_outputs(self):
        # This model has a custom attention output shape AND config flags, let's skip those checks
        old_has_attentions = self.has_attentions
        self.has_attentions = False
        super().test_greedy_generate_dict_outputs()
        self.has_attentions = old_has_attentions

    def test_group_beam_search_generate_dict_output(self):
        # This model has a custom attention output shape AND config flags, let's skip those checks
        old_has_attentions = self.has_attentions
        self.has_attentions = False
        super().test_group_beam_search_generate_dict_output()
        self.has_attentions = old_has_attentions

    def test_sample_generate_dict_output(self):
        # This model has a custom attention output shape AND config flags, let's skip those checks
        old_has_attentions = self.has_attentions
        self.has_attentions = False
        super().test_sample_generate_dict_output()
        self.has_attentions = old_has_attentions

    @unittest.skip("This model doesn't support padding")
    def test_left_padding_compatibility(self):
        pass


@slow
class RWKVIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "RWKV/rwkv-4-169m-pile"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def test_simple_generate(self):
        expected_output = "Hello my name is Jasmine and I am a newbie to the"
        model = RwkvForCausalLM.from_pretrained(self.model_id).to(torch_device)

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(torch_device)
        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    def test_simple_generate_bf16(self):
        expected_output = "Hello my name is Jasmine and I am a newbie to the"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(torch_device)
        model = RwkvForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)

        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)
