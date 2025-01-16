# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import math
import unittest
from typing import Dict, List, Tuple
from unittest.util import safe_repr

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, FalconMambaConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        FalconMambaForCausalLM,
        FalconMambaModel,
    )
    from transformers.cache_utils import MambaCache


# Copied from transformers.tests.models.mamba.MambaModelTester with Mamba->FalconMamba,mamba->falcon_mamba
class FalconMambaModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.tie_word_embeddings = tie_word_embeddings

    # Ignore copy
    def get_large_model_config(self):
        return FalconMambaConfig.from_pretrained("tiiuae/falcon-mamba-7b")

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = ids_tensor([self.batch_size, self.seq_length], 1)

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
            attention_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        return FalconMambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            activation_function=self.hidden_act,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_falcon_mamba_model(self, config, input_ids, *args):
        config.output_hidden_states = True
        model = FalconMambaModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def create_and_check_causal_lm(self, config, input_ids, *args):
        model = FalconMambaForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_state_equivalency(self, config, input_ids, *args):
        model = FalconMambaModel(config=config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids)
        output_whole = outputs.last_hidden_state

        outputs = model(
            input_ids[:, :-1],
            use_cache=True,
            cache_position=torch.arange(0, config.conv_kernel, device=input_ids.device),
        )
        output_one = outputs.last_hidden_state

        # Using the state computed on the first inputs, we will get the same output
        outputs = model(
            input_ids[:, -1:],
            use_cache=True,
            cache_params=outputs.cache_params,
            cache_position=torch.arange(config.conv_kernel, config.conv_kernel + 1, device=input_ids.device),
        )
        output_two = outputs.last_hidden_state

        self.parent.assertTrue(torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5))
        # TODO the orignal mamba does not support decoding more than 1 token neither do we

    def create_and_check_falcon_mamba_cached_slow_forward_and_backwards(
        self, config, input_ids, *args, gradient_checkpointing=False
    ):
        model = FalconMambaModel(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # create cache
        cache = model(input_ids, use_cache=True).cache_params
        cache.reset()

        # use cache
        token_emb = model.embeddings(input_ids)
        outputs = model.layers[0].mixer.slow_forward(
            token_emb, cache, cache_position=torch.arange(0, config.conv_kernel, device=input_ids.device)
        )

        loss = torch.log(1 + torch.abs(outputs.sum()))
        self.parent.assertEqual(loss.shape, ())
        self.parent.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.hidden_size))
        loss.backward()

    def create_and_check_falcon_mamba_lm_head_forward_and_backwards(
        self, config, input_ids, *args, gradient_checkpointing=False
    ):
        model = FalconMambaForCausalLM(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            attention_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
# Copied from transformers.tests.models.mamba.MambaModelTest with Mamba->Falcon,mamba->falcon_mamba,FalconMambaCache->MambaCache
class FalconMambaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FalconMambaModel, FalconMambaForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (FalconMambaForCausalLM,) if is_torch_available() else ()
    has_attentions = False  # FalconMamba does not support attentions
    fx_compatible = False  # FIXME let's try to support this @ArthurZucker
    test_torchscript = False  # FIXME let's try to support this @ArthurZucker
    test_missing_keys = False
    test_model_parallel = False
    test_pruning = False
    test_head_masking = False  # FalconMamba does not have attention heads
    pipeline_model_mapping = (
        {"feature-extraction": FalconMambaModel, "text-generation": FalconMambaForCausalLM}
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = FalconMambaModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=FalconMambaConfig, n_embd=37, common_properties=["hidden_size", "num_hidden_layers"]
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

    @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["cache_params"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = torch.nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    def test_falcon_mamba_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_falcon_mamba_model(*config_and_inputs)

    def test_falcon_mamba_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm(*config_and_inputs)

    def test_state_equivalency(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_state_equivalency(*config_and_inputs)

    def test_falcon_mamba_cached_slow_forward_and_backwards(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_falcon_mamba_cached_slow_forward_and_backwards(*config_and_inputs)

    def test_falcon_mamba_lm_head_forward_and_backwards(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_falcon_mamba_lm_head_forward_and_backwards(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                if "dt_proj.bias" in name:
                    dt = torch.exp(
                        torch.tensor([0, 1]) * (math.log(config.time_step_max) - math.log(config.time_step_min))
                        + math.log(config.time_step_min)
                    ).clamp(min=config.time_step_floor)
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    if param.requires_grad:
                        self.assertTrue(param.data.max().item() <= inv_dt[1])
                        self.assertTrue(param.data.min().item() >= inv_dt[0])
                elif "A_log" in name:
                    A = torch.arange(1, config.state_size + 1, dtype=torch.float32)[None, :]
                    torch.testing.assert_close(param.data, torch.log(A), rtol=1e-5, atol=1e-5)
                elif "D" in name:
                    if param.requires_grad:
                        # check if it's a ones like
                        torch.testing.assert_close(param.data, torch.ones_like(param.data), rtol=1e-5, atol=1e-5)

    @slow
    # Ignore copy
    def test_model_from_pretrained(self):
        model = FalconMambaModel.from_pretrained(
            "tiiuae/falcon-mamba-7b", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.assertIsNotNone(model)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, MambaCache):  # MODIFIED PART START
                        recursive_check(tuple_object.conv_states, dict_object.conv_states)
                        recursive_check(tuple_object.ssm_states, dict_object.ssm_states)
                    elif isinstance(tuple_object, (List, Tuple)):  # MODIFIED PART END
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(tuple_object, dict_object, atol=1e-5),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})


@require_torch
@require_torch_accelerator
@slow
class FalconMambaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "tiiuae/falcon-mamba-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.text = "Hello today"

    def test_generation_bf16(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")

        inputs = self.tokenizer(self.text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        self.assertEqual(
            self.tokenizer.batch_decode(out, skip_special_tokens=False)[0],
            "Hello today I am going to show you how to make a simple and easy to make paper plane.\nStep",
        )

    @require_bitsandbytes
    def test_generation_4bit(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=quantization_config)

        inputs = self.tokenizer(self.text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        self.assertEqual(
            self.tokenizer.batch_decode(out, skip_special_tokens=False)[0],
            """Hello today I'm going to talk about the "C" in the "C-I-""",
        )

    def test_generation_torch_compile(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
        model = torch.compile(model)

        inputs = self.tokenizer(self.text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        self.assertEqual(
            self.tokenizer.batch_decode(out, skip_special_tokens=False)[0],
            "Hello today I am going to show you how to make a simple and easy to make paper plane.\nStep",
        )

    def test_batched_generation(self):
        model_id = "tiiuae/falcon-mamba-7b"
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.pad_token_id = tok.eos_token_id

        texts = ["Hello today", "Hello my name is Younes and today"]

        EXPECTED_OUTPUT = [
            "Hello today I'm going to show you how to make a 3D model of a house.\n",
            "Hello my name is Younes and today I will be talking about the topic of “The importance of the internet in our life”.\n",
        ]

        inputs = tok(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(torch_device)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=0, torch_dtype=torch.bfloat16)

        out = model.generate(**inputs, max_new_tokens=20)
        out = tok.batch_decode(out, skip_special_tokens=True)

        self.assertListEqual(out, EXPECTED_OUTPUT)

        # We test the same generations with inputs_embeds
        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(inputs.pop("input_ids"))

        inputs["inputs_embeds"] = inputs_embeds
        out = model.generate(**inputs, max_new_tokens=20)
        out = tok.batch_decode(out, skip_special_tokens=True)

        self.assertListEqual(out, EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_training_kernel(self):
        model_id = "tiiuae/falcon-mamba-7b"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        text = "Hello today"

        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            logits = torch.argmax(model(**inputs).logits, dim=-1)

        out_no_training = tokenizer.batch_decode(logits)

        model.train()
        lm_logits = model(**inputs).logits
        next_token = torch.argmax(lm_logits, dim=-1)

        out_training = tokenizer.batch_decode(next_token)

        # Just verify backward works
        loss = (1 - lm_logits).mean()
        loss.backward()

        self.assertEqual(out_training, out_no_training)
