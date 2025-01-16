# coding=utf-8
# Copyright 2018 LXMERT Authors, The Hugging Face Team.
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

from transformers import LxmertConfig, is_tf_available, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        LxmertForPreTraining,
        LxmertForQuestionAnswering,
        LxmertModel,
    )


if is_tf_available():
    import tensorflow as tf


class LxmertModelTester:
    def __init__(
        self,
        parent,
        vocab_size=300,
        hidden_size=28,
        num_attention_heads=2,
        num_labels=2,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_qa_labels=30,
        num_object_labels=16,
        num_attr_labels=4,
        num_visual_features=10,
        l_layers=2,
        x_layers=1,
        r_layers=1,
        visual_feat_dim=128,
        visual_pos_dim=4,
        visual_loss_normalizer=6.67,
        seq_length=20,
        batch_size=4,
        is_training=True,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        visual_obj_loss=True,
        visual_attr_loss=True,
        visual_feat_loss=True,
        use_token_type_ids=True,
        use_lang_mask=True,
        output_attentions=False,
        output_hidden_states=False,
        scope=None,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.num_qa_labels = num_qa_labels
        self.num_object_labels = num_object_labels
        self.num_attr_labels = num_attr_labels
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.visual_loss_normalizer = visual_loss_normalizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_lang_mask = use_lang_mask
        self.task_matched = task_matched
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_qa = task_qa
        self.visual_obj_loss = visual_obj_loss
        self.visual_attr_loss = visual_attr_loss
        self.visual_feat_loss = visual_feat_loss
        self.num_visual_features = num_visual_features
        self.use_token_type_ids = use_token_type_ids
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.scope = scope
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}

    def prepare_config_and_inputs(self):
        output_attentions = self.output_attentions
        input_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=self.vocab_size)
        visual_feats = torch.rand(self.batch_size, self.num_visual_features, self.visual_feat_dim, device=torch_device)
        bounding_boxes = torch.rand(self.batch_size, self.num_visual_features, 4, device=torch_device)

        input_mask = None
        if self.use_lang_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        obj_labels = None
        if self.task_obj_predict:
            obj_labels = {}
        if self.visual_attr_loss and self.task_obj_predict:
            obj_labels["attr"] = (
                ids_tensor([self.batch_size, self.num_visual_features], self.num_attr_labels),
                ids_tensor([self.batch_size, self.num_visual_features], self.num_attr_labels),
            )
        if self.visual_feat_loss and self.task_obj_predict:
            obj_labels["feat"] = (
                ids_tensor(
                    [self.batch_size, self.num_visual_features, self.visual_feat_dim], self.num_visual_features
                ),
                ids_tensor([self.batch_size, self.num_visual_features], self.num_visual_features),
            )
        if self.visual_obj_loss and self.task_obj_predict:
            obj_labels["obj"] = (
                ids_tensor([self.batch_size, self.num_visual_features], self.num_object_labels),
                ids_tensor([self.batch_size, self.num_visual_features], self.num_object_labels),
            )
        ans = None
        if self.task_qa:
            ans = ids_tensor([self.batch_size], self.num_qa_labels)
        masked_lm_labels = None
        if self.task_mask_lm:
            masked_lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        matched_label = None
        if self.task_matched:
            matched_label = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return (
            config,
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids,
            input_mask,
            obj_labels,
            masked_lm_labels,
            matched_label,
            ans,
            output_attentions,
        )

    def get_config(self):
        return LxmertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_labels=self.num_labels,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            num_qa_labels=self.num_qa_labels,
            num_object_labels=self.num_object_labels,
            num_attr_labels=self.num_attr_labels,
            l_layers=self.l_layers,
            x_layers=self.x_layers,
            r_layers=self.r_layers,
            visual_feat_dim=self.visual_feat_dim,
            visual_pos_dim=self.visual_pos_dim,
            visual_loss_normalizer=self.visual_loss_normalizer,
            task_matched=self.task_matched,
            task_mask_lm=self.task_mask_lm,
            task_obj_predict=self.task_obj_predict,
            task_qa=self.task_qa,
            visual_obj_loss=self.visual_obj_loss,
            visual_attr_loss=self.visual_attr_loss,
            visual_feat_loss=self.visual_feat_loss,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    def create_and_check_lxmert_model(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_label,
        ans,
        output_attentions,
    ):
        model = LxmertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=not output_attentions,
        )
        result = model(input_ids, visual_feats, bounding_boxes, return_dict=False)
        result = model(input_ids, visual_feats, bounding_boxes, return_dict=True)

        self.parent.assertEqual(result.language_output.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(
            result.vision_output.shape, (self.batch_size, self.num_visual_features, self.hidden_size)
        )
        self.parent.assertEqual(result.pooled_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_lxmert_for_question_answering(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_label,
        ans,
        output_attentions,
    ):
        model = LxmertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            labels=ans,
            output_attentions=output_attentions,
        )
        result = model(input_ids, visual_feats, bounding_boxes, labels=ans)
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            labels=ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            labels=ans,
            output_attentions=not output_attentions,
        )

        self.parent.assertEqual(result.question_answering_score.shape, (self.batch_size, self.num_qa_labels))

    def create_and_check_lxmert_for_pretraining(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_label,
        ans,
        output_attentions,
    ):
        model = LxmertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            obj_labels=obj_labels,
            matched_label=matched_label,
            ans=ans,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            output_attentions=not output_attentions,
            return_dict=False,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            obj_labels=obj_labels,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            matched_label=matched_label,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            ans=ans,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            obj_labels=obj_labels,
            matched_label=matched_label,
            ans=ans,
            output_attentions=not output_attentions,
        )

        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def resize_lxmert_num_qa_labels(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_label,
        ans,
        output_attentions,
    ):
        start_labels = config.num_qa_labels
        num_large_labels = config.num_qa_labels * 2
        num_small_labels = int(config.num_qa_labels * 2)
        less_labels_ans = ids_tensor([self.batch_size], num_small_labels)
        more_labels_ans = ids_tensor([self.batch_size], num_large_labels)
        model_pretrain = LxmertForPreTraining(config=config).to(torch_device)
        model_qa = LxmertForQuestionAnswering(config=config).to(torch_device)
        config.num_labels = num_small_labels
        end_labels = config.num_labels

        result_pretrain = model_pretrain(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            ans=ans,
        )

        result_qa = model_qa(
            input_ids,
            visual_feats,
            bounding_boxes,
            labels=ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )

        model_pretrain.resize_num_qa_labels(num_small_labels)
        model_qa.resize_num_qa_labels(num_small_labels)

        result_pretrain_less = model_pretrain(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            ans=less_labels_ans,
        )

        result_qa_less = model_qa(
            input_ids,
            visual_feats,
            bounding_boxes,
            labels=less_labels_ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )

        model_pretrain.resize_num_qa_labels(num_large_labels)
        model_qa.resize_num_qa_labels(num_large_labels)

        result_pretrain_more = model_pretrain(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            ans=more_labels_ans,
        )

        result_qa_more = model_qa(
            input_ids,
            visual_feats,
            bounding_boxes,
            labels=more_labels_ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )

        model_qa_labels = model_qa.num_qa_labels

        self.parent.assertNotEqual(start_labels, end_labels)
        self.parent.assertNotEqual(model_qa_labels, start_labels)
        self.parent.assertEqual(result_qa.question_answering_score.shape, (self.batch_size, start_labels))
        self.parent.assertEqual(result_pretrain.question_answering_score.shape, (self.batch_size, start_labels))
        self.parent.assertEqual(result_qa_less.question_answering_score.shape, (self.batch_size, num_small_labels))
        self.parent.assertEqual(
            result_pretrain_less.question_answering_score.shape, (self.batch_size, num_small_labels)
        )
        self.parent.assertEqual(result_qa_more.question_answering_score.shape, (self.batch_size, num_large_labels))
        self.parent.assertEqual(
            result_pretrain_more.question_answering_score.shape, (self.batch_size, num_large_labels)
        )

    def prepare_config_and_inputs_for_common(self, return_obj_labels=False):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids,
            input_mask,
            obj_labels,
            masked_lm_labels,
            matched_label,
            ans,
            output_attentions,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "visual_feats": visual_feats,
            "visual_pos": bounding_boxes,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }

        if return_obj_labels:
            inputs_dict["obj_labels"] = obj_labels
        else:
            config.task_obj_predict = False

        return config, inputs_dict


@require_torch
class LxmertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LxmertModel, LxmertForPreTraining, LxmertForQuestionAnswering) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": LxmertModel, "question-answering": LxmertForQuestionAnswering}
        if is_torch_available()
        else {}
    )

    fx_compatible = True
    test_head_masking = False
    test_pruning = False
    test_torchscript = False

    # overwrite function because qa models takes different input label shape
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                # special case for models like BERT that use multi-loss training for PreTraining
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = LxmertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LxmertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_lxmert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_model(*config_and_inputs)

    def test_lxmert_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_for_question_answering(*config_and_inputs)

    def test_lxmert_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_for_pretraining(*config_and_inputs)

    def test_lxmert_question_answering_labels_resize(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.resize_lxmert_num_qa_labels(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "unc-nlp/lxmert-base-uncased"
        model = LxmertModel.from_pretrained(model_name)
        model.to(torch_device)
        self.assertIsNotNone(model)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            language_attentions, vision_attentions, cross_encoder_attentions = (outputs[-3], outputs[-2], outputs[-1])

            self.assertEqual(len(language_attentions), self.model_tester.num_hidden_layers["language"])
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers["vision"])
            self.assertEqual(len(cross_encoder_attentions), self.model_tester.num_hidden_layers["cross_encoder"])

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            language_attentions, vision_attentions, cross_encoder_attentions = (outputs[-3], outputs[-2], outputs[-1])
            self.assertEqual(len(language_attentions), self.model_tester.num_hidden_layers["language"])
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers["vision"])
            self.assertEqual(len(cross_encoder_attentions), self.model_tester.num_hidden_layers["cross_encoder"])

            attentions = [language_attentions, vision_attentions, cross_encoder_attentions]
            attention_shapes = [
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_visual_features,
                    self.model_tester.num_visual_features,
                ],
                [self.model_tester.num_attention_heads, encoder_key_length, self.model_tester.num_visual_features],
            ]

            for attention, attention_shape in zip(attentions, attention_shapes):
                self.assertListEqual(list(attention[0].shape[-3:]), attention_shape)
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # 2 hidden states were added
            self.assertEqual(out_len + 2, len(outputs))

            language_attentions, vision_attentions, cross_encoder_attentions = (outputs[-3], outputs[-2], outputs[-1])
            self.assertEqual(len(language_attentions), self.model_tester.num_hidden_layers["language"])
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers["vision"])
            self.assertEqual(len(cross_encoder_attentions), self.model_tester.num_hidden_layers["cross_encoder"])

            attentions = [language_attentions, vision_attentions, cross_encoder_attentions]
            attention_shapes = [
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_visual_features,
                    self.model_tester.num_visual_features,
                ],
                [self.model_tester.num_attention_heads, encoder_key_length, self.model_tester.num_visual_features],
            ]

            for attention, attention_shape in zip(attentions, attention_shapes):
                self.assertListEqual(list(attention[0].shape[-3:]), attention_shape)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            language_hidden_states, vision_hidden_states = outputs[-2], outputs[-1]

            self.assertEqual(len(language_hidden_states), self.model_tester.num_hidden_layers["language"] + 1)
            self.assertEqual(len(vision_hidden_states), self.model_tester.num_hidden_layers["vision"] + 1)

            seq_length = self.model_tester.seq_length
            num_visual_features = self.model_tester.num_visual_features

            self.assertListEqual(
                list(language_hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(vision_hidden_states[0].shape[-2:]),
                [num_visual_features, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        hidden_states_lang = outputs.language_hidden_states[0]
        attentions_lang = outputs.language_attentions[0]

        hidden_states_vision = outputs.vision_hidden_states[0]
        attentions_vision = outputs.vision_attentions[0]

        hidden_states_lang.retain_grad()
        attentions_lang.retain_grad()
        hidden_states_vision.retain_grad()
        attentions_vision.retain_grad()

        outputs.language_output.flatten()[0].backward(retain_graph=True)
        outputs.vision_output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states_lang.grad)
        self.assertIsNotNone(attentions_vision.grad)
        self.assertIsNotNone(hidden_states_vision.grad)
        self.assertIsNotNone(attentions_vision.grad)

    def prepare_tf_inputs_from_pt_inputs(self, pt_inputs_dict):
        tf_inputs_dict = {}
        for key, value in pt_inputs_dict.items():
            # skip key that does not exist in tf
            if isinstance(value, dict):
                tf_inputs_dict[key] = self.prepare_pt_inputs_from_tf_inputs(value)
            elif isinstance(value, (list, tuple)):
                tf_inputs_dict[key] = (self.prepare_pt_inputs_from_tf_inputs(iter_value) for iter_value in value)
            elif isinstance(value, bool):
                tf_inputs_dict[key] = value
            elif key == "input_values":
                tf_inputs_dict[key] = tf.convert_to_tensor(value.cpu().numpy(), dtype=tf.float32)
            elif key == "pixel_values":
                tf_inputs_dict[key] = tf.convert_to_tensor(value.cpu().numpy(), dtype=tf.float32)
            elif key == "input_features":
                tf_inputs_dict[key] = tf.convert_to_tensor(value.cpu().numpy(), dtype=tf.float32)
            # other general float inputs
            elif value.is_floating_point():
                tf_inputs_dict[key] = tf.convert_to_tensor(value.cpu().numpy(), dtype=tf.float32)
            else:
                tf_inputs_dict[key] = tf.convert_to_tensor(value.cpu().numpy(), dtype=tf.int32)

        return tf_inputs_dict

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


@require_torch
class LxmertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        input_ids = torch.tensor([[101, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 102]])
        num_visual_features = 10
        _, visual_feats = np.random.seed(0), np.random.rand(1, num_visual_features, model.config.visual_feat_dim)
        _, visual_pos = np.random.seed(0), np.random.rand(1, num_visual_features, 4)
        visual_feats = torch.as_tensor(visual_feats, dtype=torch.float32)
        visual_pos = torch.as_tensor(visual_pos, dtype=torch.float32)
        output = model(input_ids, visual_feats=visual_feats, visual_pos=visual_pos)[0]
        expected_shape = torch.Size([1, 11, 768])
        self.assertEqual(expected_shape, output.shape)
        expected_slice = torch.tensor(
            [[[0.2417, -0.9807, 0.1480], [1.2541, -0.8320, 0.5112], [1.4070, -1.1052, 0.6990]]]
        )

        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
