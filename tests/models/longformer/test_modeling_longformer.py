# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import LongformerConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        LongformerForMaskedLM,
        LongformerForMultipleChoice,
        LongformerForQuestionAnswering,
        LongformerForSequenceClassification,
        LongformerForTokenClassification,
        LongformerModel,
    )
    from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


class LongformerModelTester:
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
        scope=None,
        attention_window=4,
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
        self.attention_window = attention_window

        # `ModelTesterMixin.test_attention_outputs` is expecting attention tensors to be of size
        # [num_attention_heads, encoder_seq_length, encoder_key_length], but LongformerSelfAttention
        # returns attention of shape [num_attention_heads, encoder_seq_length, self.attention_window + 1]
        # because its local attention only attends to `self.attention_window + 1` locations
        # (assuming no token with global attention, otherwise the last dimension of attentions
        # is x + self.attention_window + 1, where x is the number of tokens with global attention)
        self.key_length = self.attention_window + 2

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
        return LongformerConfig(
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
            initializer_range=self.initializer_range,
            attention_window=self.attention_window,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_attention_mask_determinism(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        output_with_mask = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        output_without_mask = model(input_ids)["last_hidden_state"]
        self.parent.assertTrue(torch.allclose(output_with_mask[0, 0, :5], output_without_mask[0, 0, :5], atol=1e-4))

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_global_attention_mask(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        global_attention_mask = input_mask.clone()
        global_attention_mask[:, input_mask.shape[-1] // 2] = 0
        global_attention_mask = global_attention_mask.to(torch_device)

        result = model(
            input_ids,
            attention_mask=input_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
        )
        result = model(input_ids, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask)
        result = model(input_ids, global_attention_mask=global_attention_mask)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            global_attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = LongformerForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = LongformerForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = LongformerForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            global_attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

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
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, -1] = 1

        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "global_attention_mask": global_attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_question_answering(self):
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

        # Replace sep_token_id by some random id
        input_ids[input_ids == config.sep_token_id] = torch.randint(0, config.vocab_size, (1,)).item()
        # Make sure there are exactly three sep_token_id
        input_ids[:, -3:] = config.sep_token_id
        input_mask = torch.ones_like(input_ids)

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels


@require_torch
class LongformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_pruning = False  # pruning is not supported
    test_torchscript = False

    all_model_classes = (
        (
            LongformerModel,
            LongformerForMaskedLM,
            LongformerForSequenceClassification,
            LongformerForQuestionAnswering,
            LongformerForTokenClassification,
            LongformerForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": LongformerModel,
            "fill-mask": LongformerForMaskedLM,
            "question-answering": LongformerForQuestionAnswering,
            "text-classification": LongformerForSequenceClassification,
            "token-classification": LongformerForTokenClassification,
            "zero-shot": LongformerForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    # Need to use `0.6` instead of `0.5` for `test_disk_offload`
    model_split_percents = [0.6, 0.7, 0.9]

    # TODO: Fix the failed tests
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
        if (
            pipeline_test_case_name == "QAPipelineTests"
            and tokenizer_name is not None
            and not tokenizer_name.endswith("Fast")
        ):
            # `QAPipelineTests` fails for a few models when the slower tokenizer are used.
            # (The slower tokenizers were never used for pipeline tests before the pipeline testing rework)
            # TODO: check (and possibly fix) the `QAPipelineTests` with slower tokenizer
            return True

        return False

    def setUp(self):
        self.model_tester = LongformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask_determinism(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_attention_mask_determinism(*config_and_inputs)

    def test_model_global_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_global_attention_mask(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_question_answering()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    @unittest.skip(reason="Longformer cannot keep gradients in attention or hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        return

    @unittest.skip(reason="LongFormer calculates global attn only when attn_mask has non-zero elements")
    def test_batching_equivalence(self):
        return


@require_torch
@require_sentencepiece
@require_tokenizers
class LongformerModelIntegrationTest(unittest.TestCase):
    def _get_hidden_states(self):
        return torch.tensor(
            [
                [
                    [
                        4.98332758e-01,
                        2.69175139e00,
                        -7.08081422e-03,
                        1.04915401e00,
                        -1.83476661e00,
                        7.67220476e-01,
                        2.98580543e-01,
                        2.84803992e-02,
                    ],
                    [
                        -7.58357372e-01,
                        4.20635998e-01,
                        -4.04739919e-02,
                        1.59924145e-01,
                        2.05135748e00,
                        -1.15997978e00,
                        5.37166397e-01,
                        2.62873606e-01,
                    ],
                    [
                        -1.69438001e00,
                        4.17574660e-01,
                        -1.49196962e00,
                        -1.76483717e00,
                        -1.94566312e-01,
                        -1.71183858e00,
                        7.72903565e-01,
                        -1.11557056e00,
                    ],
                    [
                        5.44028163e-01,
                        2.05466114e-01,
                        -3.63045868e-01,
                        2.41865062e-01,
                        3.20348382e-01,
                        -9.05611176e-01,
                        -1.92690727e-01,
                        -1.19917547e00,
                    ],
                ]
            ],
            dtype=torch.float32,
            device=torch_device,
        )

    def test_diagonalize(self):
        hidden_states = self._get_hidden_states()
        hidden_states = hidden_states.reshape((1, 8, 4))  # set seq length = 8, hidden dim = 4
        chunked_hidden_states = LongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        window_overlap_size = chunked_hidden_states.shape[2]
        self.assertTrue(window_overlap_size == 4)

        padded_hidden_states = LongformerSelfAttention._pad_and_diagonalize(chunked_hidden_states)

        self.assertTrue(padded_hidden_states.shape[-1] == chunked_hidden_states.shape[-1] + window_overlap_size - 1)

        # first row => [0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000]
        torch.testing.assert_close(
            padded_hidden_states[0, 0, 0, :4], chunked_hidden_states[0, 0, 0], rtol=1e-3, atol=1e-3
        )
        self.assertTrue(
            torch.allclose(
                padded_hidden_states[0, 0, 0, 4:],
                torch.zeros((3,), device=torch_device, dtype=torch.float32),
                atol=1e-3,
            )
        )
        # last row => [0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629]
        torch.testing.assert_close(
            padded_hidden_states[0, 0, -1, 3:], chunked_hidden_states[0, 0, -1], rtol=1e-3, atol=1e-3
        )
        self.assertTrue(
            torch.allclose(
                padded_hidden_states[0, 0, -1, :3],
                torch.zeros((3,), device=torch_device, dtype=torch.float32),
                atol=1e-3,
            )
        )

    def test_pad_and_transpose_last_two_dims(self):
        hidden_states = self._get_hidden_states()
        self.assertEqual(hidden_states.shape, (1, 4, 8))
        padding = (0, 0, 0, 1)

        padded_hidden_states = LongformerSelfAttention._pad_and_transpose_last_two_dims(hidden_states, padding)
        self.assertEqual(padded_hidden_states.shape, (1, 8, 5))

        expected_added_dim = torch.zeros((5,), device=torch_device, dtype=torch.float32)
        torch.testing.assert_close(expected_added_dim, padded_hidden_states[0, -1, :], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            hidden_states[0, -1, :], padded_hidden_states.view(1, -1)[0, 24:32], rtol=1e-6, atol=1e-6
        )

    def test_chunk(self):
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = hidden_states.reshape((batch_size, seq_length, hidden_size))

        chunked_hidden_states = LongformerSelfAttention._chunk(hidden_states, window_overlap=2)

        # expected slices across chunk and seq length dim
        expected_slice_along_seq_length = torch.tensor(
            [0.4983, -0.7584, -1.6944], device=torch_device, dtype=torch.float32
        )
        expected_slice_along_chunk = torch.tensor(
            [0.4983, -1.8348, -0.7584, 2.0514], device=torch_device, dtype=torch.float32
        )

        torch.testing.assert_close(
            chunked_hidden_states[0, :, 0, 0], expected_slice_along_seq_length, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(chunked_hidden_states[0, 0, :, 0], expected_slice_along_chunk, rtol=1e-3, atol=1e-3)
        self.assertEqual(chunked_hidden_states.shape, (1, 3, 4, 4))

    def test_mask_invalid_locations(self):
        hidden_states = self._get_hidden_states()

        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = hidden_states.reshape((batch_size, seq_length, hidden_size))
        chunked_hidden_states = LongformerSelfAttention._chunk(hidden_states, window_overlap=2)

        hid_states_1 = chunked_hidden_states.clone()
        LongformerSelfAttention._mask_invalid_locations(hid_states_1, 1)
        self.assertTrue(torch.isinf(hid_states_1).sum().item() == 8)

        hid_states_2 = chunked_hidden_states.clone()
        LongformerSelfAttention._mask_invalid_locations(hid_states_2, 2)
        self.assertTrue(torch.isinf(hid_states_2).sum().item() == 24)

        hid_states_3 = chunked_hidden_states.clone()[:, :, :, :3]
        LongformerSelfAttention._mask_invalid_locations(hid_states_3, 2)
        self.assertTrue(torch.isinf(hid_states_3).sum().item() == 24)

        hid_states_4 = chunked_hidden_states.clone()[:, :, 2:, :]
        LongformerSelfAttention._mask_invalid_locations(hid_states_4, 2)
        self.assertTrue(torch.isinf(hid_states_4).sum().item() == 12)

    def test_layer_local_attn(self):
        model = LongformerModel.from_pretrained("patrickvonplaten/longformer-random-tiny")
        model.eval()
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = self._get_hidden_states()
        batch_size, seq_length, hidden_size = hidden_states.size()
        attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=torch_device)
        attention_mask[:, -2:] = -10000

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        output_hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
        )[0]

        self.assertEqual(output_hidden_states.shape, (1, 4, 8))
        self.assertTrue(
            torch.allclose(
                output_hidden_states[0, 1],
                torch.tensor(
                    [0.0019, 0.0122, -0.0171, -0.0256, -0.0300, 0.0173, -0.0115, 0.0048],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

    def test_layer_global_attn(self):
        model = LongformerModel.from_pretrained("patrickvonplaten/longformer-random-tiny")
        model.eval()
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = torch.cat([self._get_hidden_states(), self._get_hidden_states() - 0.5], dim=0)
        batch_size, seq_length, hidden_size = hidden_states.size()
        attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=torch_device)

        # create attn mask
        attention_mask[0, -2:] = 10000.0
        attention_mask[0, -1:] = -10000.0
        attention_mask[1, 1:] = 10000.0

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        output_hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
        )[0]

        self.assertEqual(output_hidden_states.shape, (2, 4, 8))

        self.assertTrue(
            torch.allclose(
                output_hidden_states[0, 2],
                torch.tensor(
                    [-0.0651, -0.0393, 0.0309, -0.0342, -0.0066, -0.0155, -0.0209, -0.0494],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

        self.assertTrue(
            torch.allclose(
                output_hidden_states[1, -2],
                torch.tensor(
                    [-0.0405, -0.0384, 0.0396, -0.0374, -0.0341, 0.0136, 0.0014, -0.0571],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

    def test_layer_attn_probs(self):
        model = LongformerModel.from_pretrained("patrickvonplaten/longformer-random-tiny")
        model.eval()
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = torch.cat([self._get_hidden_states(), self._get_hidden_states() - 0.5], dim=0)
        batch_size, seq_length, hidden_size = hidden_states.size()
        attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=torch_device)

        # create attn mask
        attention_mask[0, -2:] = 10000.0
        attention_mask[0, -1:] = -10000.0
        attention_mask[1, 1:] = 10000.0

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        output_hidden_states, local_attentions, global_attentions = layer(
            hidden_states,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=True,
        )

        self.assertEqual(local_attentions.shape, (2, 4, 2, 8))
        self.assertEqual(global_attentions.shape, (2, 2, 3, 4))

        # All tokens with global attention have weight 0 in local attentions.
        self.assertTrue(torch.all(local_attentions[0, 2:4, :, :] == 0))
        self.assertTrue(torch.all(local_attentions[1, 1:4, :, :] == 0))

        # The weight of all tokens with local attention must sum to 1.
        self.assertTrue(torch.all(torch.abs(global_attentions[0, :, :2, :].sum(dim=-1) - 1) < 1e-6))
        self.assertTrue(torch.all(torch.abs(global_attentions[1, :, :1, :].sum(dim=-1) - 1) < 1e-6))

        self.assertTrue(
            torch.allclose(
                local_attentions[0, 0, 0, :],
                torch.tensor(
                    [0.3328, 0.0000, 0.0000, 0.0000, 0.0000, 0.3355, 0.3318, 0.0000],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

        self.assertTrue(
            torch.allclose(
                local_attentions[1, 0, 0, :],
                torch.tensor(
                    [0.2492, 0.2502, 0.2502, 0.0000, 0.0000, 0.2505, 0.0000, 0.0000],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

        # All the global attention weights must sum to 1.
        self.assertTrue(torch.all(torch.abs(global_attentions.sum(dim=-1) - 1) < 1e-6))

        self.assertTrue(
            torch.allclose(
                global_attentions[0, 0, 1, :],
                torch.tensor(
                    [0.2500, 0.2500, 0.2500, 0.2500],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

        self.assertTrue(
            torch.allclose(
                global_attentions[1, 0, 0, :],
                torch.tensor(
                    [0.2497, 0.2500, 0.2499, 0.2504],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )

    @slow
    def test_inference_no_head(self):
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        model.to(torch_device)

        # 'Hello world!'
        input_ids = torch.tensor([[0, 20920, 232, 328, 1437, 2]], dtype=torch.long, device=torch_device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        output = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]

        expected_output_slice = torch.tensor([0.0549, 0.1087, -0.1119, -0.0368, 0.0250], device=torch_device)
        torch.testing.assert_close(output[0, 0, -5:], expected_output_slice, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output_without_mask[0, 0, -5:], expected_output_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_no_head_long(self):
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        model.to(torch_device)

        # 'Hello world! ' repeated 1000 times
        input_ids = torch.tensor(
            [[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device
        )  # long input

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[:, [1, 4, 21]] = 1  # Set global attention on a few random positions

        output = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)[0]

        expected_output_sum = torch.tensor(74585.8594, device=torch_device)
        expected_output_mean = torch.tensor(0.0243, device=torch_device)
        torch.testing.assert_close(output.sum(), expected_output_sum, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output.mean(), expected_output_mean, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_masked_lm_long(self):
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        model.to(torch_device)

        # 'Hello world! ' repeated 1000 times
        input_ids = torch.tensor(
            [[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device
        )  # long input
        input_ids = input_ids.to(torch_device)

        loss, prediction_scores = model(input_ids, labels=input_ids).to_tuple()

        expected_loss = torch.tensor(0.0074, device=torch_device)
        expected_prediction_scores_sum = torch.tensor(-6.1048e08, device=torch_device)
        expected_prediction_scores_mean = torch.tensor(-3.0348, device=torch_device)

        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(prediction_scores.sum(), expected_prediction_scores_sum, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(prediction_scores.mean(), expected_prediction_scores_mean, rtol=1e-4, atol=1e-4)
