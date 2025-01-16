# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GPTNeoX model."""

import unittest

from parameterized import parameterized

from transformers import AutoTokenizer, GPTNeoXConfig, is_torch_available, set_seed
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GPTNeoXForCausalLM,
        GPTNeoXForQuestionAnswering,
        GPTNeoXForSequenceClassification,
        GPTNeoXForTokenClassification,
        GPTNeoXModel,
    )
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding


class GPTNeoXModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
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
        scope=None,
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
        self.scope = scope
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, token_labels

    def get_config(self):
        return GPTNeoXConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_decoder(self):
        config, input_ids, input_mask, token_labels = self.prepare_config_and_inputs()

        config.is_decoder = True

        return config, input_ids, input_mask, token_labels

    def create_and_check_model(self, config, input_ids, input_mask):
        model = GPTNeoXModel(config=config)
        model.to(torch_device)
        model.eval()
        _ = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, input_mask):
        config.add_cross_attention = True
        model = GPTNeoXModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, token_labels):
        model = GPTNeoXForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = GPTNeoXForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = GPTNeoXForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = GPTNeoXForTokenClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, input_mask):
        config.is_decoder = True
        model = GPTNeoXForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True)
        output_from_no_past = output_from_no_past["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
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

    def create_and_check_cached_forward_with_and_without_attention_mask(self, config, input_ids, *args):
        # Relevant issue: https://github.com/huggingface/transformers/issues/31943
        model = GPTNeoXModel(config)
        model.to(torch_device)
        model.eval()

        # We want this for SDPA, eager works with a `None` attention mask
        assert (
            model.config._attn_implementation == "sdpa"
        ), "This test assumes the model to have the SDPA implementation for its attention calculations."

        # Prepare cache and non_cache input, needs a full attention mask
        cached_len = input_ids.shape[-1] // 2
        input_mask = torch.ones(size=input_ids.size()).to(torch_device)
        cache_inputs = {"input_ids": input_ids[:, :cached_len], "attention_mask": input_mask[:, :cached_len]}
        non_cache_inputs = {"input_ids": input_ids[:, cached_len:], "attention_mask": input_mask}

        # Cached forward once with the attention mask provided and the other time without it (which should assume full attention)
        cache_outputs = model(**cache_inputs)
        full_outputs_with_attention_mask = model(
            **non_cache_inputs, past_key_values=cache_outputs.past_key_values
        ).last_hidden_state
        full_outputs_without_attention_mask = model(
            non_cache_inputs["input_ids"], past_key_values=cache_outputs.past_key_values
        ).last_hidden_state

        self.parent.assertTrue(
            torch.allclose(full_outputs_with_attention_mask, full_outputs_without_attention_mask, atol=1e-5)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, token_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GPTNeoXModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            GPTNeoXModel,
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (GPTNeoXForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTNeoXModel,
            "question-answering": GPTNeoXForQuestionAnswering,
            "text-classification": GPTNeoXForSequenceClassification,
            "text-generation": GPTNeoXForCausalLM,
            "token-classification": GPTNeoXForTokenClassification,
            "zero-shot": GPTNeoXForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = GPTNeoXModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTNeoXConfig, hidden_size=64, num_attention_heads=8)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, input_mask)

    def test_model_as_decoder(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(config, input_ids, input_mask)

    def test_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(config, input_ids, input_mask)

    def test_decoder_model_past_large_inputs(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(config, input_ids, input_mask)

    def test_model_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_model_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_model_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_model_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_cached_forward_with_and_without_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cached_forward_with_and_without_attention_mask(*config_and_inputs)

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",)])
    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_model_rope_scaling_from_config with Llama->GPTNeoX
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = GPTNeoXModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = GPTNeoXModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = GPTNeoXRotaryEmbedding(config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        linear_scaling_rope = GPTNeoXRotaryEmbedding(config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        ntk_scaling_rope = GPTNeoXRotaryEmbedding(config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())


@require_torch
class GPTNeoXLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        for checkpointing in [True, False]:
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
            # See: https://github.com/huggingface/transformers/pull/24193
            expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    @slow
    def test_lm_generate_flex_attn_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        for checkpointing in [True, False]:
            model = GPTNeoXForCausalLM.from_pretrained(
                "EleutherAI/pythia-410m-deduped", attn_implementation="flex_attention"
            )
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
            # See: https://github.com/huggingface/transformers/pull/24193
            expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    def pythia_integration_test(self):
        model_name_or_path = "EleutherAI/pythia-70m"
        model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(torch_device)
        EXPECTED_LOGITS = torch.tensor([1069.0000,  228.7500, 1072.0000, 1072.0000, 1069.0000, 1068.0000, 1068.0000, 1071.0000, 1071.0000, 1071.0000, 1073.0000, 1070.0000, 1071.0000, 1075.0000, 1073.0000, 1075.0000, 1074.0000, 1069.0000, 1072.0000, 1071.0000, 1071.0000, 1071.0000, 1070.0000, 1069.0000, 1069.0000, 1069.0000, 1070.0000, 1075.0000, 1073.0000, 1074.0000])  # fmt: skip
        input_ids = [29, 93, 303, 64, 5478, 49651, 10394, 187, 34, 12939, 875]
        # alternative: tokenizer('<|im_start|>system\nA chat between')
        input_ids = torch.as_tensor(input_ids)[None].to(torch_device)
        outputs = model(input_ids)["logits"][:, -1][0, :30]
        torch.testing.assert_close(EXPECTED_LOGITS, outputs, rtol=1e-5, atol=1e-5)
