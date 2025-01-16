# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LUKE model."""

import unittest

from transformers import LukeConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        LukeForEntityClassification,
        LukeForEntityPairClassification,
        LukeForEntitySpanClassification,
        LukeForMaskedLM,
        LukeForMultipleChoice,
        LukeForQuestionAnswering,
        LukeForSequenceClassification,
        LukeForTokenClassification,
        LukeModel,
        LukeTokenizer,
    )


class LukeModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        entity_length=3,
        mention_length=5,
        use_attention_mask=True,
        use_token_type_ids=True,
        use_entity_ids=True,
        use_entity_attention_mask=True,
        use_entity_token_type_ids=True,
        use_entity_position_ids=True,
        use_labels=True,
        vocab_size=99,
        entity_vocab_size=10,
        entity_emb_size=6,
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
        num_entity_classification_labels=9,
        num_entity_pair_classification_labels=6,
        num_entity_span_classification_labels=4,
        use_entity_aware_attention=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.entity_length = entity_length
        self.mention_length = mention_length
        self.use_attention_mask = use_attention_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_entity_ids = use_entity_ids
        self.use_entity_attention_mask = use_entity_attention_mask
        self.use_entity_token_type_ids = use_entity_token_type_ids
        self.use_entity_position_ids = use_entity_position_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.entity_emb_size = entity_emb_size
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
        self.num_entity_classification_labels = num_entity_classification_labels
        self.num_entity_pair_classification_labels = num_entity_pair_classification_labels
        self.num_entity_span_classification_labels = num_entity_span_classification_labels
        self.scope = scope
        self.use_entity_aware_attention = use_entity_aware_attention

        self.encoder_seq_length = seq_length
        self.key_length = seq_length
        self.num_hidden_states_types = 2  # hidden_states and entity_hidden_states

    def prepare_config_and_inputs(self):
        # prepare words
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        # prepare entities
        entity_ids = ids_tensor([self.batch_size, self.entity_length], self.entity_vocab_size)

        entity_attention_mask = None
        if self.use_entity_attention_mask:
            entity_attention_mask = random_attention_mask([self.batch_size, self.entity_length])

        entity_token_type_ids = None
        if self.use_token_type_ids:
            entity_token_type_ids = ids_tensor([self.batch_size, self.entity_length], self.type_vocab_size)

        entity_position_ids = None
        if self.use_entity_position_ids:
            entity_position_ids = ids_tensor(
                [self.batch_size, self.entity_length, self.mention_length], self.mention_length
            )

        sequence_labels = None
        token_labels = None
        choice_labels = None
        entity_labels = None
        entity_classification_labels = None
        entity_pair_classification_labels = None
        entity_span_classification_labels = None

        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

            entity_labels = ids_tensor([self.batch_size, self.entity_length], self.entity_vocab_size)

            entity_classification_labels = ids_tensor([self.batch_size], self.num_entity_classification_labels)
            entity_pair_classification_labels = ids_tensor(
                [self.batch_size], self.num_entity_pair_classification_labels
            )
            entity_span_classification_labels = ids_tensor(
                [self.batch_size, self.entity_length], self.num_entity_span_classification_labels
            )

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            entity_ids,
            entity_attention_mask,
            entity_token_type_ids,
            entity_position_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            entity_labels,
            entity_classification_labels,
            entity_pair_classification_labels,
            entity_span_classification_labels,
        )

    def get_config(self):
        return LukeConfig(
            vocab_size=self.vocab_size,
            entity_vocab_size=self.entity_vocab_size,
            entity_emb_size=self.entity_emb_size,
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
            use_entity_aware_attention=self.use_entity_aware_attention,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        model = LukeModel(config=config)
        model.to(torch_device)
        model.eval()
        # test with words + entities
        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
        )
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(
            result.entity_last_hidden_state.shape, (self.batch_size, self.entity_length, self.hidden_size)
        )

        # test with words only
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_labels = self.num_entity_classification_labels
        model = LukeForMaskedLM(config)
        model.to(torch_device)
        model.eval()

        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            labels=token_labels,
            entity_labels=entity_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        if entity_ids is not None:
            self.parent.assertEqual(
                result.entity_logits.shape, (self.batch_size, self.entity_length, self.entity_vocab_size)
            )
        else:
            self.parent.assertIsNone(result.entity_logits)

    def create_and_check_for_entity_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_labels = self.num_entity_classification_labels
        model = LukeForEntityClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            labels=entity_classification_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_entity_classification_labels))

    def create_and_check_for_entity_pair_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_labels = self.num_entity_pair_classification_labels
        model = LukeForEntityClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            labels=entity_pair_classification_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_entity_pair_classification_labels))

    def create_and_check_for_entity_span_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_labels = self.num_entity_span_classification_labels
        model = LukeForEntitySpanClassification(config)
        model.to(torch_device)
        model.eval()

        entity_start_positions = ids_tensor([self.batch_size, self.entity_length], self.seq_length)
        entity_end_positions = ids_tensor([self.batch_size, self.entity_length], self.seq_length)

        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            entity_start_positions=entity_start_positions,
            entity_end_positions=entity_end_positions,
            labels=entity_span_classification_labels,
        )
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.entity_length, self.num_entity_span_classification_labels)
        )

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        model = LukeForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_labels = self.num_labels
        model = LukeForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_labels = self.num_labels
        model = LukeForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        entity_ids,
        entity_attention_mask,
        entity_token_type_ids,
        entity_position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        entity_labels,
        entity_classification_labels,
        entity_pair_classification_labels,
        entity_span_classification_labels,
    ):
        config.num_choices = self.num_choices
        model = LukeForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_entity_ids = entity_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_entity_token_type_ids = (
            entity_token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        )
        multiple_choice_entity_attention_mask = (
            entity_attention_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        )
        multiple_choice_entity_position_ids = (
            entity_position_ids.unsqueeze(1).expand(-1, self.num_choices, -1, -1).contiguous()
        )
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_attention_mask,
            token_type_ids=multiple_choice_token_type_ids,
            entity_ids=multiple_choice_entity_ids,
            entity_attention_mask=multiple_choice_entity_attention_mask,
            entity_token_type_ids=multiple_choice_entity_token_type_ids,
            entity_position_ids=multiple_choice_entity_position_ids,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            entity_ids,
            entity_attention_mask,
            entity_token_type_ids,
            entity_position_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            entity_labels,
            entity_classification_labels,
            entity_pair_classification_labels,
            entity_span_classification_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "entity_ids": entity_ids,
            "entity_token_type_ids": entity_token_type_ids,
            "entity_attention_mask": entity_attention_mask,
            "entity_position_ids": entity_position_ids,
        }
        return config, inputs_dict


@require_torch
class LukeModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            LukeModel,
            LukeForMaskedLM,
            LukeForEntityClassification,
            LukeForEntityPairClassification,
            LukeForEntitySpanClassification,
            LukeForQuestionAnswering,
            LukeForSequenceClassification,
            LukeForTokenClassification,
            LukeForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": LukeModel,
            "fill-mask": LukeForMaskedLM,
            "question-answering": LukeForQuestionAnswering,
            "text-classification": LukeForSequenceClassification,
            "token-classification": LukeForTokenClassification,
            "zero-shot": LukeForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = True
    test_head_masking = True

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
        if pipeline_test_case_name in ["QAPipelineTests", "ZeroShotClassificationPipelineTests"]:
            return True

        return False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        entity_inputs_dict = {k: v for k, v in inputs_dict.items() if k.startswith("entity")}
        inputs_dict = {k: v for k, v in inputs_dict.items() if not k.startswith("entity")}

        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if model_class == LukeForMultipleChoice:
            entity_inputs_dict = {
                k: v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                if v.ndim == 2
                else v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1, -1).contiguous()
                for k, v in entity_inputs_dict.items()
            }
        inputs_dict.update(entity_inputs_dict)

        if model_class == LukeForEntitySpanClassification:
            inputs_dict["entity_start_positions"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.entity_length), dtype=torch.long, device=torch_device
            )
            inputs_dict["entity_end_positions"] = torch.ones(
                (self.model_tester.batch_size, self.model_tester.entity_length), dtype=torch.long, device=torch_device
            )

        if return_labels:
            if model_class in (
                LukeForEntityClassification,
                LukeForEntityPairClassification,
                LukeForSequenceClassification,
                LukeForMultipleChoice,
            ):
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class == LukeForEntitySpanClassification:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.entity_length),
                    dtype=torch.long,
                    device=torch_device,
                )
            elif model_class == LukeForTokenClassification:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length),
                    dtype=torch.long,
                    device=torch_device,
                )
            elif model_class == LukeForMaskedLM:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length),
                    dtype=torch.long,
                    device=torch_device,
                )
                inputs_dict["entity_labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.entity_length),
                    dtype=torch.long,
                    device=torch_device,
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = LukeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LukeConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "studio-ousia/luke-base"
        model = LukeModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_masked_lm_with_word_only(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config_and_inputs = (*config_and_inputs[:4], *((None,) * len(config_and_inputs[4:])))
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
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

    def test_for_entity_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_entity_classification(*config_and_inputs)

    def test_for_entity_pair_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_entity_pair_classification(*config_and_inputs)

    def test_for_entity_span_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_entity_span_classification(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_length = self.model_tester.seq_length
        entity_length = self.model_tester.entity_length
        key_length = seq_length + entity_length

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_length + entity_length, key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = self.model_tester.num_hidden_states_types
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_length + entity_length, key_length],
            )

    def test_entity_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            entity_hidden_states = outputs.entity_hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(entity_hidden_states), expected_num_layers)

            entity_length = self.model_tester.entity_length

            self.assertListEqual(
                list(entity_hidden_states[0].shape[-2:]),
                [entity_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_entity_hidden_states(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        entity_hidden_states = outputs.entity_hidden_states[0]
        entity_hidden_states.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(entity_hidden_states.grad)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass


@require_torch
class LukeModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_base_model(self):
        model = LukeModel.from_pretrained("studio-ousia/luke-base").eval()
        model.to(torch_device)

        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_classification")
        text = (
            "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped"
            " the new world number one avoid a humiliating second- round exit at Wimbledon ."
        )
        span = (39, 42)
        encoding = tokenizer(text, entity_spans=[span], add_prefix_space=True, return_tensors="pt")

        # move all values to device
        for key, value in encoding.items():
            encoding[key] = encoding[key].to(torch_device)

        outputs = model(**encoding)

        # Verify word hidden states
        expected_shape = torch.Size((1, 42, 768))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.0037, 0.1368, -0.0091], [0.1099, 0.3329, -0.1095], [0.0765, 0.5335, 0.1179]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # Verify entity hidden states
        expected_shape = torch.Size((1, 1, 768))
        self.assertEqual(outputs.entity_last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor([[0.1457, 0.1044, 0.0174]]).to(torch_device)
        torch.testing.assert_close(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_large_model(self):
        model = LukeModel.from_pretrained("studio-ousia/luke-large").eval()
        model.to(torch_device)

        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large", task="entity_classification")
        text = (
            "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped"
            " the new world number one avoid a humiliating second- round exit at Wimbledon ."
        )
        span = (39, 42)
        encoding = tokenizer(text, entity_spans=[span], add_prefix_space=True, return_tensors="pt")

        # move all values to device
        for key, value in encoding.items():
            encoding[key] = encoding[key].to(torch_device)

        outputs = model(**encoding)

        # Verify word hidden states
        expected_shape = torch.Size((1, 42, 1024))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.0133, 0.0865, 0.0095], [0.3093, -0.2576, -0.7418], [-0.1720, -0.2117, -0.2869]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # Verify entity hidden states
        expected_shape = torch.Size((1, 1, 1024))
        self.assertEqual(outputs.entity_last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor([[0.0466, -0.0106, -0.0179]]).to(torch_device)
        torch.testing.assert_close(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
