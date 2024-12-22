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


import unittest
from typing import Dict, List, Tuple

from parameterized import parameterized

from transformers import AutoTokenizer, Mamba2Config, is_torch_available
from transformers.testing_utils import require_read_token, require_torch, require_torch_gpu, slow, torch_device
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Mamba2ForCausalLM,
        Mamba2Model,
    )
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2Mixer


class Mamba2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        num_heads=8,
        n_groups=8,
        state_size=2,
        head_dim=8,
        conv_kernel=4,
        chunk_size=8,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=False,
    ):
        self.parent = parent
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
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

    def get_large_model_config(self):
        return Mamba2Config.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # Only left padding is valid
        attention_mask = torch.ones(size=(self.batch_size, self.seq_length), device=input_ids.device, dtype=torch.long)
        attention_mask[0, :1] = 0

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config(
            gradient_checkpointing=gradient_checkpointing,
        )

        return (
            config,
            input_ids,
            attention_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self, gradient_checkpointing=False):
        return Mamba2Config(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            n_groups=self.n_groups,
            state_size=self.state_size,
            conv_kernel=self.conv_kernel,
            chunk_size=self.chunk_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
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

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            _,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict

    def create_and_check_mamba2_caching(self, config, input_ids, attention_mask, *args):
        model = Mamba2Model(config=config)
        model.to(torch_device)
        model.eval()

        output_whole = model(input_ids, attention_mask=attention_mask).last_hidden_state

        outputs = model(
            input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
            cache_position=torch.arange(0, config.conv_kernel, device=input_ids.device),
        )
        output_one = outputs.last_hidden_state

        # Using the state computed on the first inputs, we will get the same output
        outputs = model(
            input_ids[:, -1:],
            attention_mask=attention_mask[:, -1:],
            use_cache=True,
            cache_params=outputs.cache_params,
            cache_position=torch.arange(config.conv_kernel, config.conv_kernel + 1, device=input_ids.device),
        )
        output_two = outputs.last_hidden_state

        self.parent.assertTrue(
            torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-3, rtol=1e-3)
        )

    def create_and_check_mamba2_slow_vs_fast_forward(self, config, input_ids, *args, gradient_checkpointing=False):
        model = Mamba2Model(config)
        model.eval()

        if not (is_mamba_2_ssm_available() and is_causal_conv1d_available()):
            self.parent.skipTest(
                "This test needs the Mamba2 fast path. Skipping as the necessary packages have not been found."
            )
        if torch_device != "cuda":
            self.parent.skipTest("This test needs the Mamba2 fast path. Skipping as we need a cuda capable device.")

        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        token_emb = model.embeddings(input_ids)
        outputs_fast = model.layers[0].mixer.cuda_kernels_forward(token_emb)
        outputs_slow = model.layers[0].mixer.torch_forward(token_emb)

        self.parent.assertTrue(torch.allclose(outputs_fast, outputs_slow, atol=1e-3, rtol=1e-3))


@require_torch
class Mamba2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Mamba2Model, Mamba2ForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (Mamba2ForCausalLM,) if is_torch_available() else ()
    has_attentions = False  # Mamba does not support attentions
    fx_compatible = False  # FIXME let's try to support this @molbap
    test_torchscript = False  # FIXME I think this should be doable @molbap @ArthurZucker
    test_missing_keys = False
    test_model_parallel = False
    test_pruning = False
    test_head_masking = False  # Mamba does not have attention heads

    pipeline_model_mapping = (
        {"feature-extraction": Mamba2Model, "text-generation": Mamba2ForCausalLM} if is_torch_available() else {}
    )

    def setUp(self):
        self.model_tester = Mamba2ModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Mamba2Config, n_embd=37, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def test_mamba2_caching(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_caching(*config_and_inputs)

    def test_mamba2_slow_vs_fast_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_slow_vs_fast_forward(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                if "D" in name:
                    if param.requires_grad:
                        # check if it's a ones like
                        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data), atol=1e-5, rtol=1e-5))

    @unittest.skip(reason="Mamba 2 weights are not tied")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="A large mamba2 would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, Mamba2Cache):  # MODIFIED PART START
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
@slow
@require_read_token
class Mamba2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "mistralai/Mamba-Codestral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_slow=True, legacy=False)
        self.prompt = ("[INST]Write a hello world program in C++.",)

    @require_read_token
    @parameterized.expand(
        [
            (torch_device,),
        ]
    )
    @slow
    @require_torch
    def test_simple_generate(self, device):
        """
        Simple generate test to avoid regressions.
        Note: state-spaces (cuda) implementation and pure torch implementation
        have irreconciliable differences as of now, which will cause this test to fail
        in an environment with state-spaces installed.
        """
        tokenizer = self.tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        model.to(device)
        input_ids = tokenizer("[INST]Write a hello world program in C++.[/INST]", return_tensors="pt")["input_ids"].to(
            device
        )

        out = model.generate(input_ids, do_sample=False, use_cache=True, max_new_tokens=30)
        output_sentence = tokenizer.decode(out[0])
        ground_truth_sentence = """<s>[INST]Write a hello world program in C++.[/INST] Sure, here is a simple "Hello, World!" program in C++:\n\n```cpp\n#include <iostream>\n\n"""
        self.assertEqual(output_sentence, ground_truth_sentence)

    @require_read_token
    @slow
    @require_torch_gpu
    def test_batched_equivalence_with_cache(self):
        """
        Verifies that batched generation matches individual generation.
        Important because of the specific caching mechanism + statefulness of mamba model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batched generation
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest").to(torch_device)
        batched_gen = model.generate(**tokenized_prompts, max_new_tokens=30, use_cache=True)
        batched_output = tokenizer.batch_decode(batched_gen, skip_special_tokens=True)

        # individual generation

        for index_gen, individual_prompt in enumerate(prompt):
            inputs = tokenizer(individual_prompt, return_tensors="pt", padding="longest").to(torch_device)
            individual_gen = model.generate(**inputs, max_new_tokens=30, use_cache=True)
            individual_output = tokenizer.batch_decode(individual_gen, skip_special_tokens=True)[0]
            self.assertEqual(individual_output[:100], batched_output[index_gen][:100])

    @require_read_token
    @slow
    @require_torch_gpu
    def test_batched_equivalence_without_cache(self):
        """
        Verifies that batched generation matches individual generation without cache.
        Important because of the specific caching mechanism + statefulness of mamba model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batched generation
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest").to(torch_device)
        batched_gen = model.generate(**tokenized_prompts, max_new_tokens=30, use_cache=True)
        batched_output = tokenizer.batch_decode(batched_gen, skip_special_tokens=True)

        # individual generation

        for index_gen, individual_prompt in enumerate(prompt):
            inputs = tokenizer(individual_prompt, return_tensors="pt", padding="longest").to(torch_device)
            individual_gen = model.generate(**inputs, max_new_tokens=30, use_cache=True)
            individual_output = tokenizer.batch_decode(individual_gen, skip_special_tokens=True)[0]
            self.assertEqual(individual_output[:100], batched_output[index_gen][:100])

    @slow
    @require_torch_gpu
    def test_mamba2_mixer_train_vs_eval_equivalence(self):
        # Based on https://github.com/sustcsonglin/flash-linear-attention/issues/63
        # Credit to zhixuan-lin

        B, T, D = 4, 512, 768
        dtype = torch.bfloat16
        config = Mamba2Config(num_heads=24, head_dim=64, hidden_size=768, expand=2, n_groups=1)

        torch.manual_seed(42)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            with torch.no_grad():
                mixer = Mamba2Mixer(config, layer_idx=0).to("cuda")
                hidden_states = torch.rand(size=(B, T, D), dtype=dtype, device="cuda")

                mixer.train()
                out_train = mixer(hidden_states)

                mixer.eval()
                out_eval = mixer(hidden_states)

                self.assertTrue(torch.allclose(out_train, out_eval, atol=1e-3))
