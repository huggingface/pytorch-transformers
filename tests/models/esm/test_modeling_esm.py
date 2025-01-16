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
"""Testing suite for the PyTorch ESM model."""

import unittest

from transformers import EsmConfig, is_torch_available
from transformers.testing_utils import TestCasePlus, require_bitsandbytes, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import EsmForMaskedLM, EsmForSequenceClassification, EsmForTokenClassification, EsmModel
    from transformers.models.esm.modeling_esm import (
        EsmEmbeddings,
        create_position_ids_from_input_ids,
    )


# copied from tests.test_modeling_roberta
class EsmModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=33,
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return EsmConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            pad_token_id=1,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = EsmModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = EsmForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = EsmForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_forward_and_backwards(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        gradient_checkpointing=False,
    ):
        model = EsmForMaskedLM(config)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to(torch_device)
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class EsmModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_mismatched_shapes = False

    all_model_classes = (
        (
            EsmForMaskedLM,
            EsmModel,
            EsmForSequenceClassification,
            EsmForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "feature-extraction": EsmModel,
            "fill-mask": EsmForMaskedLM,
            "text-classification": EsmForSequenceClassification,
            "token-classification": EsmForTokenClassification,
            "zero-shot": EsmForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_sequence_classification_problem_types = True
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = EsmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EsmConfig, hidden_size=37)

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

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_esm_gradient_checkpointing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs, gradient_checkpointing=True)

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/esm2_t6_8M_UR50D"
        model = EsmModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_create_position_ids_respects_padding_index(self):
        """This is a regression test for https://github.com/huggingface/transformers/issues/1761

        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is EsmEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        model = EsmEmbeddings(config=config)

        input_ids = torch.as_tensor([[12, 31, 13, model.padding_idx]])
        expected_positions = torch.as_tensor(
            [
                [
                    0 + model.padding_idx + 1,
                    1 + model.padding_idx + 1,
                    2 + model.padding_idx + 1,
                    model.padding_idx,
                ]
            ]
        )
        position_ids = create_position_ids_from_input_ids(input_ids, model.padding_idx)
        self.assertEqual(position_ids.shape, expected_positions.shape)
        self.assertTrue(torch.all(torch.eq(position_ids, expected_positions)))

    def test_create_position_ids_from_inputs_embeds(self):
        """This is a regression test for https://github.com/huggingface/transformers/issues/1761

        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is EsmEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        embeddings = EsmEmbeddings(config=config)

        inputs_embeds = torch.empty(2, 4, 30)
        expected_single_positions = [
            0 + embeddings.padding_idx + 1,
            1 + embeddings.padding_idx + 1,
            2 + embeddings.padding_idx + 1,
            3 + embeddings.padding_idx + 1,
        ]
        expected_positions = torch.as_tensor([expected_single_positions, expected_single_positions])
        position_ids = embeddings.create_position_ids_from_inputs_embeds(inputs_embeds)
        self.assertEqual(position_ids.shape, expected_positions.shape)
        self.assertTrue(torch.all(torch.eq(position_ids, expected_positions)))

    @unittest.skip(reason="Esm does not support embedding resizing")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip(reason="Esm does not support embedding resizing")
    def test_resize_tokens_embeddings(self):
        pass


@slow
@require_torch
class EsmModelIntegrationTest(TestCasePlus):
    def test_inference_masked_lm(self):
        with torch.no_grad():
            model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
            model.eval()
            input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
            output = model(input_ids)[0]

            vocab_size = 33

            expected_shape = torch.Size((1, 6, vocab_size))
            self.assertEqual(output.shape, expected_shape)

            expected_slice = torch.tensor(
                [[[8.9215, -10.5898, -6.4671], [-6.3967, -13.9114, -1.1212], [-7.7812, -13.9516, -3.7406]]]
            )
            torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def test_inference_no_head(self):
        with torch.no_grad():
            model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            model.eval()

            input_ids = torch.tensor([[0, 6, 4, 13, 5, 4, 16, 12, 11, 7, 2]])
            output = model(input_ids)[0]
            # compare the actual values for a slice.
            expected_slice = torch.tensor(
                [[[0.1444, 0.5413, 0.3248], [0.3034, 0.0053, 0.3108], [0.3228, -0.2499, 0.3415]]]
            )
            torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @require_bitsandbytes
    def test_inference_bitsandbytes(self):
        model = EsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D", load_in_8bit=True)

        input_ids = torch.tensor([[0, 6, 4, 13, 5, 4, 16, 12, 11, 7, 2]])
        # Just test if inference works
        with torch.no_grad():
            _ = model(input_ids)[0]

        model = EsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D", load_in_4bit=True)

        input_ids = torch.tensor([[0, 6, 4, 13, 5, 4, 16, 12, 11, 7, 2]])
        # Just test if inference works
        _ = model(input_ids)[0]
