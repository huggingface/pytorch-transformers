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

from transformers import AutoTokenizer, is_torch_available, xLSTMConfig
from transformers.testing_utils import require_read_token, require_torch, require_torch_gpu, slow, torch_device
from transformers.utils import is_xlstm_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        xLSTMForCausalLM,
        xLSTMModel,
    )
    from transformers.models.xlstm.modeling_xlstm import mLSTMBlock, xLSTMCache
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_2
else:
    is_torch_greater_or_equal_than_2_2 = False


class xLSTMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_heads=2,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        embedding_dim=128,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        num_blocks=2,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        chunkwise_kernel="chunkwise--native_autograd",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        tie_word_embeddings=False,
    ):
        self.parent = parent
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.tie_word_embeddings = tie_word_embeddings

        self.num_hidden_layers = self.num_blocks
        self.hidden_size = self.embedding_dim

    def get_large_model_config(self):
        cfg = xLSTMConfig.from_pretrained("NX-AI/xLSTM-7b")
        # this is needed for compatibility with generic tests
        cfg.hidden_size = cfg.embedding_dim
        cfg.num_hidden_layers = cfg.num_blocks
        return cfg

    def prepare_config_and_inputs(self, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            None,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        cfg = xLSTMConfig(
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_blocks=self.num_blocks,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            tie_word_embeddings=self.tie_word_embeddings,
        )
        # this is needed for compatibility with generic tests
        cfg.hidden_size = cfg.embedding_dim
        cfg.num_hidden_layers = cfg.num_blocks
        return cfg

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


@unittest.skipIf(
    not is_torch_greater_or_equal_than_2_2, reason="See https://github.com/huggingface/transformers/pull/24204"
)
@unittest.skipIf(not is_xlstm_available(), reason="xlstm needs to be installed for tests to work")
@require_torch
class xLSTMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (xLSTMModel, xLSTMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (xLSTMForCausalLM,) if is_torch_available() else ()
    has_attentions = False  # xLSTM does not support attentions
    fx_compatible = False  # FIXME let's try to support this @molbap
    test_torchscript = False  # FIXME I think this should be doable @molbap @ArthurZucker
    test_missing_keys = False
    test_model_parallel = False
    test_pruning = False
    test_head_masking = False  # xLSTM does not have attention heads
    # TODO: why this still fails?
    test_resize_embeddings = False

    pipeline_model_mapping = (
        {"feature-extraction": xLSTMModel, "text-generation": xLSTMForCausalLM} if is_torch_available() else {}
    )

    def setUp(self):
        self.model_tester = xLSTMModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=xLSTMConfig, n_embd=37, common_properties=["embedding_dim", "num_blocks"]
        )

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                if "D" in name:
                    if param.requires_grad:
                        # check if it's a ones like
                        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data), atol=1e-5, rtol=1e-5))

    @unittest.skip(reason="xLSTM weights are not tied")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="To fix, xLSTM cache slicing test case is an edge case")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="To fix, xLSTM cache slicing test case is an edge case")
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @unittest.skip(reason="To fix, xLSTM cache slicing test case is an edge case")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="To fix, xLSTM cache slicing is interacting with beam search")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="A large xlstm would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, xLSTMCache):
                        recursive_check(tuple_object.rnn_state, dict_object.rnn_state)
                    elif isinstance(tuple_object, (List, Tuple)):
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


@unittest.skipIf(not is_xlstm_available(), reason="xlstm needs to be installed for tests to work")
@require_torch
@slow
@require_read_token
class xLSTMIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "NX-AI/xLSTM-7b"
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

        model = xLSTMForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
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
        Important because of the specific caching mechanism + statefulness of the xLSTM model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = xLSTMForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
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
        Important because of the specific caching mechanism + statefulness of the xLSTM model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = xLSTMForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
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
    def test_xlstm_block_train_vs_eval_equivalence(self):
        # Based on https://github.com/sustcsonglin/flash-linear-attention/issues/63
        # Credit to zhixuan-lin

        B, T, D = 4, 512, 768
        dtype = torch.bfloat16
        config = xLSTMConfig(num_heads=24, head_dim=64, embedding_dim=768, expand=2, n_groups=1)

        torch.manual_seed(42)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            with torch.no_grad():
                block = mLSTMBlock(config, layer_idx=0).to("cuda")
                hidden_states = torch.rand(size=(B, T, D), dtype=dtype, device="cuda")

                block.train()
                out_train = block(hidden_states)

                block.eval()
                out_eval = block(hidden_states)

                self.assertTrue(torch.allclose(out_train, out_eval, atol=1e-3))
