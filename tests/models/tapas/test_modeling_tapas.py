# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

import copy
import unittest

import numpy as np
import pandas as pd

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TapasConfig,
    is_torch_available,
)
from transformers.models.auto import get_values
from transformers.testing_utils import require_tensorflow_probability, require_torch, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        TapasForMaskedLM,
        TapasForQuestionAnswering,
        TapasForSequenceClassification,
        TapasModel,
        TapasTokenizer,
    )
    from transformers.models.tapas.modeling_tapas import (
        IndexMap,
        ProductIndexMap,
        flatten,
        gather,
        range_index_map,
        reduce_max,
        reduce_mean,
        reduce_sum,
    )


class TapasModelTester:
    """You can also import this e.g from .test_modeling_tapas import TapasModelTester"""

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
        initializer_range=0.02,
        max_position_embeddings=512,
        type_vocab_sizes=[3, 256, 256, 2, 256, 256, 10],
        type_sequence_label_size=2,
        positive_weight=10.0,
        num_aggregation_labels=4,
        num_labels=2,
        aggregation_loss_importance=0.8,
        use_answer_as_supervision=True,
        answer_loss_importance=0.001,
        use_normalized_answer_loss=False,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function="ratio",
        cell_selection_preference=0.5,
        answer_loss_cutoff=100,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=True,
        allow_empty_column_selection=False,
        init_cell_selection_weights_to_zero=True,
        reset_position_index_per_cell=True,
        disable_per_token_loss=False,
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
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_sizes = type_vocab_sizes
        self.type_sequence_label_size = type_sequence_label_size
        self.positive_weight = positive_weight
        self.num_aggregation_labels = num_aggregation_labels
        self.num_labels = num_labels
        self.aggregation_loss_importance = aggregation_loss_importance
        self.use_answer_as_supervision = use_answer_as_supervision
        self.answer_loss_importance = answer_loss_importance
        self.use_normalized_answer_loss = use_normalized_answer_loss
        self.huber_loss_delta = huber_loss_delta
        self.temperature = temperature
        self.agg_temperature = agg_temperature
        self.use_gumbel_for_cells = use_gumbel_for_cells
        self.use_gumbel_for_agg = use_gumbel_for_agg
        self.average_approximation_function = average_approximation_function
        self.cell_selection_preference = cell_selection_preference
        self.answer_loss_cutoff = answer_loss_cutoff
        self.max_num_rows = max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).to(torch_device)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length]).to(torch_device)

        token_type_ids = []
        for type_vocab_size in self.type_vocab_sizes:
            token_type_ids.append(ids_tensor(shape=[self.batch_size, self.seq_length], vocab_size=type_vocab_size))
        token_type_ids = torch.stack(token_type_ids, dim=2).to(torch_device)

        sequence_labels = None
        token_labels = None
        labels = None
        numeric_values = None
        numeric_values_scale = None
        float_answer = None
        aggregation_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size).to(torch_device)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels).to(torch_device)
            labels = ids_tensor([self.batch_size, self.seq_length], vocab_size=2).to(torch_device)
            numeric_values = floats_tensor([self.batch_size, self.seq_length]).to(torch_device)
            numeric_values_scale = floats_tensor([self.batch_size, self.seq_length]).to(torch_device)
            float_answer = floats_tensor([self.batch_size]).to(torch_device)
            aggregation_labels = ids_tensor([self.batch_size], self.num_aggregation_labels).to(torch_device)

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            labels,
            numeric_values,
            numeric_values_scale,
            float_answer,
            aggregation_labels,
        )

    def get_config(self):
        return TapasConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_sizes=self.type_vocab_sizes,
            initializer_range=self.initializer_range,
            positive_weight=self.positive_weight,
            num_aggregation_labels=self.num_aggregation_labels,
            num_labels=self.num_labels,
            aggregation_loss_importance=self.aggregation_loss_importance,
            use_answer_as_supervision=self.use_answer_as_supervision,
            answer_loss_importance=self.answer_loss_importance,
            use_normalized_answer_loss=self.use_normalized_answer_loss,
            huber_loss_delta=self.huber_loss_delta,
            temperature=self.temperature,
            agg_temperature=self.agg_temperature,
            use_gumbel_for_cells=self.use_gumbel_for_cells,
            use_gumbel_for_agg=self.use_gumbel_for_agg,
            average_approximation_function=self.average_approximation_function,
            cell_selection_preference=self.cell_selection_preference,
            answer_loss_cutoff=self.answer_loss_cutoff,
            max_num_rows=self.max_num_rows,
            max_num_columns=self.max_num_columns,
            average_logits_per_cell=self.average_logits_per_cell,
            select_one_column=self.select_one_column,
            allow_empty_column_selection=self.allow_empty_column_selection,
            init_cell_selection_weights_to_zero=self.init_cell_selection_weights_to_zero,
            reset_position_index_per_cell=self.reset_position_index_per_cell,
            disable_per_token_loss=self.disable_per_token_loss,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        model = TapasModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        model = TapasForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        # inference: without aggregation head (SQA). Model only returns logits
        sqa_config = copy.copy(config)
        sqa_config.num_aggregation_labels = 0
        sqa_config.use_answer_as_supervision = False
        model = TapasForQuestionAnswering(config=sqa_config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

        # inference: with aggregation head (WTQ, WikiSQL-supervised). Model returns logits and aggregation logits
        model = TapasForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

        # training: can happen in 3 main ways
        # case 1: conversational (SQA)
        model = TapasForQuestionAnswering(config=sqa_config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

        # case 2: weak supervision for aggregation (WTQ)
        model = TapasForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            numeric_values=numeric_values,
            numeric_values_scale=numeric_values_scale,
            float_answer=float_answer,
        )
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

        # case 3: strong supervision for aggregation (WikiSQL-supervised)
        wikisql_config = copy.copy(config)
        wikisql_config.use_answer_as_supervision = False
        model = TapasForQuestionAnswering(config=wikisql_config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            aggregation_labels=aggregation_labels,
        )
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        config.num_labels = self.num_labels
        model = TapasForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            labels,
            numeric_values,
            numeric_values_scale,
            float_answer,
            aggregation_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class TapasModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TapasModel,
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
        )
        if is_torch_available()
        else None
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TapasModel,
            "fill-mask": TapasForMaskedLM,
            "table-question-answering": TapasForQuestionAnswering,
            "text-classification": TapasForSequenceClassification,
            "zero-shot": TapasForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {
                k: v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                if isinstance(v, torch.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }

        if return_labels:
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = torch.ones(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
            elif model_class in get_values(MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["aggregation_labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["numeric_values"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length),
                    dtype=torch.float,
                    device=torch_device,
                )
                inputs_dict["numeric_values_scale"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length),
                    dtype=torch.float,
                    device=torch_device,
                )
                inputs_dict["float_answer"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.float, device=torch_device
                )
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
        return inputs_dict

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
        return True

    def setUp(self):
        self.model_tester = TapasModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TapasConfig, dim=37)

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

    @require_tensorflow_probability
    @unittest.skip(reason="tfp is not defined even if installed. FIXME @Arthur in a followup PR!")
    def test_pt_tf_model_equivalence(self):
        pass

    @unittest.skip(reason="tfp is not defined even if installed. FIXME @Arthur in a followup PR!")
    def test_tf_from_pt_safetensors(self):
        pass


def prepare_tapas_single_inputs_for_inference():
    # Here we prepare a single table-question pair to test TAPAS inference on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
    }
    queries = "Which footballer is 33 years old?"
    table = pd.DataFrame.from_dict(data)

    return table, queries


def prepare_tapas_batch_inputs_for_inference():
    # Here we prepare a batch of 2 table-question pairs to test TAPAS inference on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
        "Number of goals": ["712", "750"],
    }
    queries = ["Which footballer is 33 years old?", "How many goals does Ronaldo have?"]
    table = pd.DataFrame.from_dict(data)

    return table, queries


def prepare_tapas_batch_inputs_for_training():
    # Here we prepare a DIFFERENT batch of 2 table-question pairs to test TAPAS training on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
        "Number of goals": ["712", "750"],
    }
    queries = ["Which footballer is 33 years old?", "What's the total number of goals?"]
    table = pd.DataFrame.from_dict(data)

    answer_coordinates = [[(0, 0)], [(0, 2), (1, 2)]]
    answer_text = [["Lionel Messi"], ["1462"]]
    float_answer = [float("NaN"), float("1462")]

    return table, queries, answer_coordinates, answer_text, float_answer


@require_torch
class TapasModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

    @slow
    def test_inference_no_head(self):
        # ideally we want to test this with the weights of tapas_inter_masklm_base_reset,
        # but since it's not straightforward to do this with the TF 1 implementation, we test it with
        # the weights of the WTQ base model (i.e. tapas_wtq_wikisql_sqa_inter_masklm_base_reset)
        model = TapasModel.from_pretrained("google/tapas-base-finetuned-wtq").to(torch_device)

        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # test the sequence output
        expected_slice = torch.tensor(
            [
                [
                    [-0.141581565, -0.599805772, 0.747186482],
                    [-0.143664181, -0.602008104, 0.749218345],
                    [-0.15169853, -0.603363097, 0.741370678],
                ]
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(outputs.last_hidden_state[:, :3, :3], expected_slice, atol=0.0005))

        # test the pooled output
        expected_slice = torch.tensor([[0.987518311, -0.970520139, -0.994303405]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.pooler_output[:, :3], expected_slice, atol=0.0005))

    @unittest.skip(reason="Model not available yet")
    def test_inference_masked_lm(self):
        pass

    # TapasForQuestionAnswering has 3 possible ways of being fine-tuned:
    # - conversational set-up (SQA)
    # - weak supervision for aggregation (WTQ, WikiSQL)
    # - strong supervision for aggregation (WikiSQL-supervised)
    # We test all of them:
    @slow
    def test_inference_question_answering_head_conversational(self):
        # note that google/tapas-base-finetuned-sqa should correspond to tapas_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-sqa").to(torch_device)

        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 21))
        self.assertEqual(logits.shape, expected_shape)

        expected_tensor = torch.tensor(
            [
                [
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -9997.22461,
                    -16.2628059,
                    -10004.082,
                    15.4330549,
                    15.4330549,
                    15.4330549,
                    -9990.42,
                    -16.3270779,
                    -16.3270779,
                    -16.3270779,
                    -16.3270779,
                    -16.3270779,
                    -10004.8506,
                ]
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(logits, expected_tensor, atol=0.015))

    @slow
    def test_inference_question_answering_head_conversational_absolute_embeddings(self):
        # note that google/tapas-small-finetuned-sqa should correspond to tapas_sqa_inter_masklm_small_reset
        # however here we test the version with absolute position embeddings
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-small-finetuned-sqa", revision="no_reset").to(
            torch_device
        )

        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 21))
        self.assertEqual(logits.shape, expected_shape)

        expected_tensor = torch.tensor(
            [
                [
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -10014.7793,
                    -18.8419304,
                    -10018.0391,
                    17.7848816,
                    17.7848816,
                    17.7848816,
                    -9981.02832,
                    -16.4005489,
                    -16.4005489,
                    -16.4005489,
                    -16.4005489,
                    -16.4005489,
                    -10013.4736,
                ]
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(logits, expected_tensor, atol=0.01))

    @slow
    def test_inference_question_answering_head_weak_supervision(self):
        # note that google/tapas-base-finetuned-wtq should correspond to tapas_wtq_wikisql_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq").to(torch_device)

        tokenizer = self.default_tokenizer
        # let's test on a batch
        table, queries = prepare_tapas_batch_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, padding="longest", return_tensors="pt")
        inputs_on_device = {k: v.to(torch_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs_on_device)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((2, 28))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [-160.375504, -160.375504, -160.375504, -10072.3965, -10070.9414, -10094.9736],
                [-9861.6123, -9861.6123, -9861.6123, -9861.6123, -9891.01172, 146.600677],
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(logits[:, -6:], expected_slice, atol=0.4))

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = torch.Size((2, 4))
        self.assertEqual(logits_aggregation.shape, expected_shape)
        expected_tensor = torch.tensor(
            [[18.8545208, -9.76614857, -6.3128891, -2.93525243], [-4.05782509, 40.0351, -5.35329962, 23.3978653]],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(logits_aggregation, expected_tensor, atol=0.001))

        # test the predicted answer coordinates and aggregation indices
        EXPECTED_PREDICTED_ANSWER_COORDINATES = [[(0, 0)], [(1, 2)]]
        EXPECTED_PREDICTED_AGGREGATION_INDICES = [0, 1]

        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits.detach().cpu(), outputs.logits_aggregation.detach().cpu()
        )

        self.assertEqual(EXPECTED_PREDICTED_ANSWER_COORDINATES, predicted_answer_coordinates)
        self.assertEqual(EXPECTED_PREDICTED_AGGREGATION_INDICES, predicted_aggregation_indices)

    @slow
    def test_training_question_answering_head_weak_supervision(self):
        # note that google/tapas-base-finetuned-wtq should correspond to tapas_wtq_wikisql_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq").to(torch_device)
        model.to(torch_device)
        # normally we should put the model in training mode but it's a pain to do this with the TF 1 implementation

        tokenizer = self.default_tokenizer
        # let's test on a batch
        table, queries, answer_coordinates, answer_text, float_answer = prepare_tapas_batch_inputs_for_training()
        inputs = tokenizer(
            table=table,
            queries=queries,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            padding="longest",
            return_tensors="pt",
        )

        # prepare data (created by the tokenizer) and move to torch_device
        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)
        token_type_ids = inputs["token_type_ids"].to(torch_device)
        labels = inputs["labels"].to(torch_device)
        numeric_values = inputs["numeric_values"].to(torch_device)
        numeric_values_scale = inputs["numeric_values_scale"].to(torch_device)

        # the answer should be prepared by the user
        float_answer = torch.FloatTensor(float_answer).to(torch_device)

        # forward pass to get loss + logits:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                numeric_values=numeric_values,
                numeric_values_scale=numeric_values_scale,
                float_answer=float_answer,
            )

        # test the loss
        loss = outputs.loss
        expected_loss = torch.tensor(3.3527612686157227e-08, device=torch_device)
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-6))

        # test the logits on the first example
        logits = outputs.logits
        expected_shape = torch.Size((2, 29))
        self.assertEqual(logits.shape, expected_shape)
        expected_slice = torch.tensor(
            [
                -160.0156,
                -160.0156,
                -160.0156,
                -160.0156,
                -160.0156,
                -10072.2266,
                -10070.8896,
                -10092.6006,
                -10092.6006,
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(logits[0, -9:], expected_slice, atol=1e-6))

        # test the aggregation logits on the second example
        logits_aggregation = outputs.logits_aggregation
        expected_shape = torch.Size((2, 4))
        self.assertEqual(logits_aggregation.shape, expected_shape)
        expected_slice = torch.tensor([-4.0538, 40.0304, -5.3554, 23.3965], device=torch_device)

        self.assertTrue(torch.allclose(logits_aggregation[1, -4:], expected_slice, atol=1e-4))

    @slow
    def test_inference_question_answering_head_strong_supervision(self):
        # note that google/tapas-base-finetuned-wikisql-supervised should correspond to tapas_wikisql_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wikisql-supervised").to(
            torch_device
        )

        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 21))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor(
            [
                [
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -18.6185989,
                    -10008.7969,
                    17.6355762,
                    17.6355762,
                    17.6355762,
                    -10002.4404,
                    -18.7111301,
                    -18.7111301,
                    -18.7111301,
                    -18.7111301,
                    -18.7111301,
                    -10007.0977,
                ]
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(logits, expected_tensor, atol=0.02))

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = torch.Size((1, 4))
        self.assertEqual(logits_aggregation.shape, expected_shape)
        expected_tensor = torch.tensor(
            [[16.5659733, -3.06624889, -2.34152961, -0.970244825]], device=torch_device
        )  # PyTorch model outputs [[16.5679, -3.0668, -2.3442, -0.9674]]

        self.assertTrue(torch.allclose(logits_aggregation, expected_tensor, atol=0.003))

    @slow
    def test_inference_classification_head(self):
        # note that google/tapas-base-finetuned-tabfact should correspond to tapas_tabfact_inter_masklm_base_reset
        model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact").to(torch_device)

        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, padding="longest", return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # test the classification logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 2))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor(
            [[0.795137286, 9.5572]], device=torch_device
        )  # Note that the PyTorch model outputs [[0.8057, 9.5281]]

        self.assertTrue(torch.allclose(outputs.logits, expected_tensor, atol=0.05))


@require_torch
class TapasUtilitiesTest(unittest.TestCase):
    def _prepare_tables(self):
        """Prepares two tables, both with three distinct rows.
        The first table has two columns:
        1.0, 2.0 | 3.0
        2.0, 0.0 | 1.0
        1.0, 3.0 | 4.0
        The second table has three columns:
        1.0 | 2.0 | 3.0
        2.0 | 0.0 | 1.0
        1.0 | 3.0 | 4.0
        Returns:
        SegmentedTensors with the tables.
        """
        values = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
                [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
            ]
        )
        row_index = IndexMap(
            indices=torch.tensor(
                [
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                ]
            ),
            num_segments=3,
            batch_dims=1,
        )
        col_index = IndexMap(
            indices=torch.tensor(
                [
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                    [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                ]
            ),
            num_segments=3,
            batch_dims=1,
        )
        return values, row_index, col_index

    def test_product_index(self):
        _, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)
        row_index_proj = cell_index.project_outer(cell_index)
        col_index_proj = cell_index.project_inner(cell_index)

        ind = cell_index.indices
        self.assertEqual(cell_index.num_segments, 9)

        # Projections should give back the original indices.
        # we use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(row_index.indices.numpy(), row_index_proj.indices.numpy())
        self.assertEqual(row_index.num_segments, row_index_proj.num_segments)
        self.assertEqual(row_index.batch_dims, row_index_proj.batch_dims)
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(col_index.indices.numpy(), col_index_proj.indices.numpy())
        self.assertEqual(col_index.batch_dims, col_index_proj.batch_dims)

        # The first and second "column" are identified in the first table.
        for i in range(3):
            self.assertEqual(ind[0, i, 0], ind[0, i, 1])
            self.assertNotEqual(ind[0, i, 0], ind[0, i, 2])

        # All rows are distinct in the first table.
        for i, i_2 in zip(range(3), range(3)):
            for j, j_2 in zip(range(3), range(3)):
                if i != i_2 and j != j_2:
                    self.assertNotEqual(ind[0, i, j], ind[0, i_2, j_2])

        # All cells are distinct in the second table.
        for i, i_2 in zip(range(3), range(3)):
            for j, j_2 in zip(range(3), range(3)):
                if i != i_2 or j != j_2:
                    self.assertNotEqual(ind[1, i, j], ind[1, i_2, j_2])

    def test_flatten(self):
        _, row_index, col_index = self._prepare_tables()
        row_index_flat = flatten(row_index)
        col_index_flat = flatten(col_index)

        shape = [3, 4, 5]
        batched_index = IndexMap(indices=torch.zeros(shape).type(torch.LongTensor), num_segments=1, batch_dims=3)
        batched_index_flat = flatten(batched_index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(
            row_index_flat.indices.numpy(), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        )
        np.testing.assert_array_equal(
            col_index_flat.indices.numpy(), [0, 0, 1, 0, 0, 1, 0, 0, 1, 3, 4, 5, 3, 4, 5, 3, 4, 5]
        )
        self.assertEqual(batched_index_flat.num_segments.numpy(), np.prod(shape))
        np.testing.assert_array_equal(batched_index_flat.indices.numpy(), range(np.prod(shape)))

    def test_range_index_map(self):
        batch_shape = [3, 4]
        num_segments = 5
        index = range_index_map(batch_shape, num_segments)

        self.assertEqual(num_segments, index.num_segments)
        self.assertEqual(2, index.batch_dims)
        indices = index.indices
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(list(indices.size()), [3, 4, 5])
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
                np.testing.assert_array_equal(indices[i, j, :].numpy(), range(num_segments))

    def test_reduce_sum(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)
        row_sum, _ = reduce_sum(values, row_index)
        col_sum, _ = reduce_sum(values, col_index)
        cell_sum, _ = reduce_sum(values, cell_index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(row_sum.numpy(), [[6.0, 3.0, 8.0], [6.0, 3.0, 8.0]])
        np.testing.assert_allclose(col_sum.numpy(), [[9.0, 8.0, 0.0], [4.0, 5.0, 8.0]])
        np.testing.assert_allclose(
            cell_sum.numpy(),
            [[3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 4.0, 4.0, 0.0], [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0]],
        )

    def test_reduce_mean(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)
        row_mean, _ = reduce_mean(values, row_index)
        col_mean, _ = reduce_mean(values, col_index)
        cell_mean, _ = reduce_mean(values, cell_index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(
            row_mean.numpy(), [[6.0 / 3.0, 3.0 / 3.0, 8.0 / 3.0], [6.0 / 3.0, 3.0 / 3.0, 8.0 / 3.0]]
        )
        np.testing.assert_allclose(col_mean.numpy(), [[9.0 / 6.0, 8.0 / 3.0, 0.0], [4.0 / 3.0, 5.0 / 3.0, 8.0 / 3.0]])
        np.testing.assert_allclose(
            cell_mean.numpy(),
            [
                [3.0 / 2.0, 3.0, 0.0, 2.0 / 2.0, 1.0, 0.0, 4.0 / 2.0, 4.0, 0.0],
                [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0],
            ],
        )

    def test_reduce_max(self):
        values = torch.as_tensor([2.0, 1.0, 0.0, 3.0])
        index = IndexMap(indices=torch.as_tensor([0, 1, 0, 1]), num_segments=2)
        maximum, _ = reduce_max(values, index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(maximum.numpy(), [2, 3])

    def test_reduce_sum_vectorized(self):
        values = torch.as_tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        index = IndexMap(indices=torch.as_tensor([[0, 0, 1]]), num_segments=2, batch_dims=0)
        sums, new_index = reduce_sum(values, index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(sums.numpy(), [3.0, 3.0])
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(new_index.indices.numpy(), [0, 1])
        np.testing.assert_array_equal(new_index.num_segments.numpy(), 2)
        np.testing.assert_array_equal(new_index.batch_dims, 0)

    def test_gather(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)

        # Compute sums and then gather. The result should have the same shape as
        # the original table and each element should contain the sum the values in
        # its cell.
        sums, _ = reduce_sum(values, cell_index)
        cell_sum = gather(sums, cell_index)
        assert cell_sum.size() == values.size()

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_allclose(
            cell_sum.numpy(),
            [[[3.0, 3.0, 3.0], [2.0, 2.0, 1.0], [4.0, 4.0, 4.0]], [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]]],
        )

    def test_gather_vectorized(self):
        values = torch.as_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        index = IndexMap(indices=torch.as_tensor([[0, 1], [1, 0]]), num_segments=2, batch_dims=1)
        result = gather(values, index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(result.numpy(), [[[1, 2], [3, 4]], [[7, 8], [5, 6]]])
