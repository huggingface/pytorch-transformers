# coding=utf-8
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Testing suite for the PyTorch Phi model."""

import unittest

from parameterized import parameterized

from transformers import PhiConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        PhiForCausalLM,
        PhiForSequenceClassification,
        PhiForTokenClassification,
        PhiModel,
    )
    from transformers.models.phi.modeling_phi import PhiRotaryEmbedding


class PhiModelTester:
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
        self.pad_token_id = pad_token_id
        self.scope = scope

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
        return PhiConfig(
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

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = PhiModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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
        model = PhiModel(config)
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
        model = PhiForCausalLM(config=config)
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
        model = PhiForCausalLM(config=config)
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
class PhiModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (PhiModel, PhiForCausalLM, PhiForSequenceClassification, PhiForTokenClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (PhiForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": PhiModel,
            "text-classification": PhiForSequenceClassification,
            "text-generation": PhiForCausalLM,
            "token-classification": PhiForTokenClassification,
            "zero-shot": PhiForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79292/workflows/fa2ba644-8953-44a6-8f67-ccd69ca6a476/jobs/1012905
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.setUp with Llama->Phi
    def setUp(self):
        self.model_tester = PhiModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PhiConfig, hidden_size=37)

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_config
    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_sequence_classification_model with Llama->Phi,llama->phi
    def test_phi_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = PhiForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_sequence_classification_model_for_single_label with Llama->Phi,llama->phi
    def test_phi_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = PhiForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_sequence_classification_model_for_multi_label with Llama->Phi,llama->phi
    def test_phi_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = PhiForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @parameterized.expand([("linear",), ("dynamic",)])
    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_model_rope_scaling_from_config with Llama->Phi
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = PhiModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = PhiModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    # Copied from tests.models.gpt_neox.test_modeling_gpt_neox.GPTNeoXModelTest.test_model_rope_scaling with GPTNeoX->Phi
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
        original_rope = PhiRotaryEmbedding(config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        linear_scaling_rope = PhiRotaryEmbedding(config).to(torch_device)
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
        ntk_scaling_rope = PhiRotaryEmbedding(config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())


@slow
@require_torch
class PhiIntegrationTest(unittest.TestCase):
    def test_model_phi_1_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhiForCausalLM.from_pretrained("microsoft/phi-1").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[2.2671,  6.7684, -2.0107, -1.2440, -1.5335, -2.3828,  6.9186,  6.4245, 3.1548,  0.9998,  0.0760,  4.4653,  4.9857,  4.2956,  1.2308, -1.4178, 0.1361,  0.5191, -0.5699, -2.2201, -3.0750, -3.9600, -4.5936, -3.7394, -2.7777,  6.1874, -0.4148, -1.5684, -0.5967,  0.2395], [1.7004,  4.0383,  0.0546,  0.4530, -0.3619, -0.9021,  1.8355,  1.3587, 1.2406,  2.5775, -0.8834,  5.1910,  4.2565,  4.1406,  3.0752, -0.9099, 1.1595,  0.0264,  0.3243, -1.1803, -1.3945, -2.1406, -3.9939, -1.4438, -2.9546,  3.9204,  1.0851, -1.0598, -1.7819, -0.4827]]).to(torch_device)  # fmt: skip

        self.assertTrue(torch.allclose(EXPECTED_OUTPUT, output[0, :2, :30], atol=1e-4, rtol=1e-4))

    def test_model_phi_1_5_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[12.2922, 13.3507,  8.6963,  9.1355,  9.3502,  9.2667, 14.2027, 13.1363, 13.5446, 11.1337,  9.9279, 16.7195, 13.0768, 14.9141, 11.9965,  8.0233, 10.3129, 10.6118, 10.0204,  9.3827,  8.8344,  8.2806,  8.0153,  8.0540, 7.0964, 16.5743, 11.1256,  9.6987, 11.4770, 10.5440], [12.3323, 14.6050,  8.9986,  8.1580,  9.5654,  6.6728, 12.5966, 12.6662, 12.2784, 11.7522,  8.2039, 16.3102, 11.2203, 13.6088, 12.0125,  9.1021, 9.8216, 10.0987,  9.0926,  8.4260,  8.8009,  7.6547,  6.8075,  7.7881, 7.4501, 15.7451, 10.5053,  8.3129, 10.0027,  9.2612]]).to(torch_device)  # fmt: skip

        self.assertTrue(torch.allclose(EXPECTED_OUTPUT, output[0, :2, :30], atol=1e-4, rtol=1e-4))

    def test_model_phi_2_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhiForCausalLM.from_pretrained("microsoft/phi-2").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[6.4830,  6.1644,  3.4055,  2.2848,  5.4654,  2.8360,  5.5975,  5.5391, 7.3101,  4.2498,  2.5913, 10.3885,  6.4359,  8.7982,  5.6534,  0.5150, 2.7498,  3.1930,  2.4334,  1.7781,  1.5613,  1.3067,  0.8291,  0.5633, 0.6522,  9.8191,  5.5771,  2.7987,  4.2845,  3.7030], [6.0642,  7.8242,  3.4634,  1.9259,  4.3169,  2.0913,  6.0446,  3.6804, 6.6736,  4.0727,  2.1791, 11.4139,  5.6795,  7.5652,  6.2039,  2.7174, 4.3266,  3.6930,  2.8058,  2.6721,  2.3047,  2.0848,  2.0972,  2.0441, 1.3160,  9.2085,  4.5557,  3.0296,  2.6045,  2.4059]]).to(torch_device)  # fmt: skip

        self.assertTrue(torch.allclose(EXPECTED_OUTPUT, output[0, :2, :30], atol=1e-3, rtol=1e-3))

    def test_phi_2_generation(self):
        model = PhiForCausalLM.from_pretrained("microsoft/phi-2")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

        inputs = tokenizer(
            "Can you help me write a formal email to a potential business partner proposing a joint venture?",
            return_tensors="pt",
            return_attention_mask=False,
        )

        outputs = model.generate(**inputs, max_new_tokens=30)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "Can you help me write a formal email to a potential business partner proposing a joint venture?\nInput: Company A: ABC Inc.\nCompany B: XYZ Ltd.\nJoint Venture: A new online platform for e-commerce"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)
