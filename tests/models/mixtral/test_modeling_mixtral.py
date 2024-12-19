# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Mixtral model."""

import unittest

import pytest

from transformers import MixtralConfig, is_torch_available
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
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
        MixtralForCausalLM,
        MixtralForQuestionAnswering,
        MixtralForSequenceClassification,
        MixtralForTokenClassification,
        MixtralModel,
    )


class MixtralModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
        router_jitter_noise=0.1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.router_jitter_noise = router_jitter_noise

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

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
        return MixtralConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            num_experts_per_tok=2,
            num_local_experts=2,
            router_jitter_noise=self.router_jitter_noise,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Mixtral
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MixtralModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Mixtral
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
        model = MixtralModel(config)
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Mixtral
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
        model = MixtralForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Mixtral
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
        model = MixtralForCausalLM(config=config)
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->Mixtral
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
# Copied from tests.models.mistral.test_modeling_mistral.MistralModelTest with Mistral->Mixtral
class MixtralModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            MixtralModel,
            MixtralForCausalLM,
            MixtralForSequenceClassification,
            MixtralForTokenClassification,
            MixtralForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (MixtralForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": MixtralModel,
            "text-classification": MixtralForSequenceClassification,
            "token-classification": MixtralForTokenClassification,
            "text-generation": MixtralForCausalLM,
            "zero-shot": MixtralForSequenceClassification,
            "question-answering": MixtralForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def setUp(self):
        self.model_tester = MixtralModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MixtralConfig, hidden_size=37)

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

    def test_torch_fx_output_loss(self):
        super().test_torch_fx_output_loss()

    def test_Mixtral_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MixtralForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Mixtral_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MixtralForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Mixtral_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = MixtralForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_token_classification_model with Llama->Mixtral,llama->Mixtral
    def test_Mixtral_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = MixtralForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @unittest.skip(reason="Mixtral buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Mixtral uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Mixtral flash attention does not support right padding")

    # Ignore copy
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_local_experts = 8
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MixtralForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_local_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of includding padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


@require_torch
class MixtralIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    @require_torch_gpu
    def test_small_model_logits(self):
        model_id = "hf-internal-testing/Mixtral-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MixtralForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(
            torch_device
        )
        # TODO: might need to tweak it in case the logits do not match on our daily runners
        # these logits have been obtained with the original megablocks impelmentation.
        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in output.
        EXPECTED_LOGITS = {
            7: torch.Tensor([[0.1670, 0.1620, 0.6094], [-0.8906, -0.1588, -0.6060], [0.1572, 0.1290, 0.7246]]).to(
                torch_device
            ),
            8: torch.Tensor([[0.1631, 0.1621, 0.6094], [-0.8906, -0.1621, -0.6094], [0.1572, 0.1270, 0.7227]]).to(
                torch_device
            ),
            9: torch.Tensor([[0.1641, 0.1621, 0.6094], [-0.8906, -0.1631, -0.6094], [0.1572, 0.1260, 0.7227]]).to(
                torch_device
            ),
        }
        with torch.no_grad():
            logits = model(dummy_input).logits

        torch.testing.assert_close(
            logits[0, :3, :3], EXPECTED_LOGITS[self.cuda_compute_capability_major_version], atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            logits[1, :3, :3], EXPECTED_LOGITS[self.cuda_compute_capability_major_version], atol=1e-3, rtol=1e-3
        )

    @slow
    @require_torch_gpu
    def test_small_model_logits_batched(self):
        model_id = "hf-internal-testing/Mixtral-tiny"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = MixtralForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(
            torch_device
        )

        # TODO: might need to tweak it in case the logits do not match on our daily runners
        #
        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_LOGITS_LEFT = {
            7: torch.Tensor(
                [[0.1750, 0.0537, 0.7007], [0.1750, 0.0537, 0.7007], [0.1750, 0.0537, 0.7007]],
            ).to(torch_device),
            8: torch.Tensor([[0.1914, 0.0508, 0.7188], [0.1953, 0.0510, 0.7227], [0.1973, 0.0562, 0.7148]]).to(
                torch_device
            ),
            9: torch.Tensor([[0.1904, 0.0513, 0.7227], [0.1943, 0.0518, 0.7227], [0.1982, 0.0557, 0.7148]]).to(
                torch_device
            ),
        }

        EXPECTED_LOGITS_LEFT_UNPADDED = {
            7: torch.Tensor(
                [[0.2212, 0.5200, -0.3816], [0.8213, -0.2313, 0.6069], [0.2664, -0.7090, 0.2468]],
            ).to(torch_device),
            8: torch.Tensor([[0.2217, 0.5195, -0.3828], [0.8203, -0.2295, 0.6055], [0.2676, -0.7109, 0.2461]]).to(
                torch_device
            ),
            9: torch.Tensor([[0.2236, 0.5195, -0.3828], [0.8203, -0.2285, 0.6055], [0.2637, -0.7109, 0.2451]]).to(
                torch_device
            ),
        }

        EXPECTED_LOGITS_RIGHT_UNPADDED = {
            7: torch.Tensor([[0.2205, 0.1232, -0.1611], [-0.3484, 0.3030, -1.0312], [0.0742, 0.7930, 0.7969]]).to(
                torch_device
            ),
            8: torch.Tensor([[0.2178, 0.1260, -0.1621], [-0.3496, 0.2988, -1.0312], [0.0693, 0.7930, 0.8008]]).to(
                torch_device
            ),
            9: torch.Tensor([[0.2197, 0.1250, -0.1611], [-0.3516, 0.3008, -1.0312], [0.0684, 0.7930, 0.8008]]).to(
                torch_device
            ),
        }

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits

        torch.testing.assert_close(
            logits[0, :3, :3], EXPECTED_LOGITS_LEFT[self.cuda_compute_capability_major_version], atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            logits[0, -3:, -3:],
            EXPECTED_LOGITS_LEFT_UNPADDED[self.cuda_compute_capability_major_version],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -3:, -3:],
            EXPECTED_LOGITS_RIGHT_UNPADDED[self.cuda_compute_capability_major_version],
            atol=1e-3,
            rtol=1e-3,
        )
