# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch DBRX model."""

import unittest

from transformers import DbrxConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import DbrxForCausalLM, DbrxModel


class DbrxModelTester:
    def __init__(
        self,
        parent,
        hidden_size=32,
        ffn_hidden_size=32,
        num_attention_heads=4,
        kv_n_heads=4,
        num_hidden_layers=5,
        max_position_embeddings=512,
        type_vocab_size=16,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        use_cache=True,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        clip_qkv=8,
        rope_theta=500000,
        attn_config_model_type="",
        emb_pdrop=0.0,
        moe_jitter_eps=0,
        moe_loss_weight=0.05,
        moe_num_experts=16,
        moe_top_k=4,
        ffn_config_model_type="",
        ffn_act_fn_name="gelu",
        initializer_range=0.02,
        output_router_logits=False,
        resid_pdrop=0.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=99,
        is_decoder=True,
        pad_token_id=0,
    ):
        # Parameters unique to testing
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.parent = parent
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels

        # attn_config params
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta
        self.attn_config_model_type = attn_config_model_type

        # ffn_config params
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.ffn_config_model_type = ffn_config_model_type
        self.ffn_act_fn_name = ffn_act_fn_name

        # Other model params
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.emb_pdrop = emb_pdrop
        self.output_router_logits = output_router_logits
        self.resid_pdrop = resid_pdrop
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.is_decoder = is_decoder
        self.pad_token_id = pad_token_id

        # Make the dictionaries
        self.ffn_config = {
            "ffn_hidden_size": self.ffn_hidden_size,
            "moe_jitter_eps": self.moe_jitter_eps,
            "moe_loss_weight": self.moe_loss_weight,
            "moe_num_experts": self.moe_num_experts,
            "moe_top_k": self.moe_top_k,
            "model_type": self.ffn_config_model_type,
            "ffn_act_fn": {"name": self.ffn_act_fn_name},
        }
        self.attn_config = {
            "clip_qkv": self.clip_qkv,
            "kv_n_heads": self.kv_n_heads,
            "model_type": self.attn_config_model_type,
            "rope_theta": self.rope_theta,
        }

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        # Behind the scenes, `DbrxConfig` maps the parameters `hidden_size`, `num_hidden_layers`,
        # `num_attention_heads`, `max_position_embeddings` to the parameters `d_model`, `n_layers`,
        # `n_heads`, `max_seq_len` respectively. We use the first group of parameters because
        # other tests expect every model to have these parameters with these specific names.
        config = DbrxConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,  # mapped to `d_model`
            num_hidden_layers=self.num_hidden_layers,  # mapped to `n_layers`
            num_attention_heads=self.num_attention_heads,  # mapped to `n_heads`
            max_position_embeddings=self.max_position_embeddings,  # mapped to `max_seq_len`
            attn_config=self.attn_config,
            ffn_config=self.ffn_config,
            resid_pdrop=self.resid_pdrop,
            emb_pdrop=self.emb_pdrop,
            use_cache=self.use_cache,
            initializer_range=self.initializer_range,
            output_router_logits=self.output_router_logits,
            is_decoder=self.is_decoder,
            pad_token_id=self.pad_token_id,
        )
        return config

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Dbrx
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DbrxModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Dbrx
    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = DbrxModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Dbrx
    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = DbrxForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = DbrxForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->Dbrx
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class DbrxModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DbrxModel, DbrxForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (DbrxForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-generation": DbrxForCausalLM} if is_torch_available() else {}
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = DbrxModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DbrxConfig, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "eitanturok/dbrx-tiny"
        model = DbrxModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Dbrx models have weight tying disabled.")
    def test_tied_weights_keys(self):
        pass

    # Offload does not work with Dbrx models because of the forward of DbrxExperts where we chunk the experts.
    # The issue is that the offloaded weights of the mlp layer are still on meta device (w1_chunked, v1_chunked, w2_chunked)
    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("Dbrx does not support `torch.compile` with `fullgraph=True`.")
    def test_generate_compile_model_forward(self):
        pass


@require_torch
class DbrxModelIntegrationTest(unittest.TestCase):
    @slow
    def test_tiny_model_logits(self):
        model = DbrxForCausalLM.from_pretrained("Rocketknight1/dbrx-tiny-random")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        vocab_size = model.vocab_size

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [
                    [-1.6300e-04, 5.0118e-04, 2.5437e-04],
                    [2.0422e-05, 2.7210e-04, -1.5125e-04],
                    [-1.5105e-04, 4.6879e-04, 3.3309e-04],
                ]
            ]
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
