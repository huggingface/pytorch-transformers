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


from __future__ import annotations

import unittest

from transformers import DebertaV2Config, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFDebertaV2ForMaskedLM,
        TFDebertaV2ForMultipleChoice,
        TFDebertaV2ForQuestionAnswering,
        TFDebertaV2ForSequenceClassification,
        TFDebertaV2ForTokenClassification,
        TFDebertaV2Model,
    )


class TFDebertaV2ModelTester:
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
        relative_attention=False,
        position_biased_input=True,
        pos_att_type="None",
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
        self.relative_attention = relative_attention
        self.position_biased_input = position_biased_input
        self.pos_att_type = pos_att_type
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

        config = DebertaV2Config(
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
            relative_attention=self.relative_attention,
            position_biased_input=self.position_biased_input,
            initializer_range=self.initializer_range,
            return_dict=True,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDebertaV2Model(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDebertaV2ForMaskedLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFDebertaV2ForSequenceClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFDebertaV2ForTokenClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDebertaV2ForQuestionAnswering(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

        result = model(inputs)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = TFDebertaV2ForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
        }
        result = model(inputs)
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
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFDebertaModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFDebertaV2Model,
            TFDebertaV2ForMaskedLM,
            TFDebertaV2ForQuestionAnswering,
            TFDebertaV2ForMultipleChoice,
            TFDebertaV2ForSequenceClassification,
            TFDebertaV2ForTokenClassification,
        )
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TFDebertaV2Model,
            "fill-mask": TFDebertaV2ForMaskedLM,
            "question-answering": TFDebertaV2ForQuestionAnswering,
            "text-classification": TFDebertaV2ForSequenceClassification,
            "token-classification": TFDebertaV2ForTokenClassification,
            "zero-shot": TFDebertaV2ForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )

    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFDebertaV2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DebertaV2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
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

    @slow
    def test_model_from_pretrained(self):
        model = TFDebertaV2Model.from_pretrained("kamalkraj/deberta-v2-xlarge")
        self.assertIsNotNone(model)

    @unittest.skip("This test was broken by the refactor in #22105, TODO @ArthurZucker")
    def test_pt_tf_model_equivalence(self):
        pass


@require_tf
class TFDeBERTaV2ModelIntegrationTest(unittest.TestCase):
    @unittest.skip(reason="Model not available yet")
    def test_inference_masked_lm(self):
        pass

    @slow
    def test_inference_no_head(self):
        model = TFDebertaV2Model.from_pretrained("kamalkraj/deberta-v2-xlarge")
        input_ids = tf.constant([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        attention_mask = tf.constant([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]

        expected_slice = tf.constant(
            [[[0.2356, 0.1948, 0.0369], [-0.1063, 0.3586, -0.5152], [-0.6399, -0.0259, -0.2525]]]
        )
        tf.debugging.assert_near(output[:, 1:4, 1:4], expected_slice, atol=1e-4)
