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
"""Testing suite for the PyTorch VisualBERT model."""

import copy
import unittest

from transformers import VisualBertConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        VisualBertForMultipleChoice,
        VisualBertForPreTraining,
        VisualBertForQuestionAnswering,
        VisualBertForRegionToPhraseAlignment,
        VisualBertForVisualReasoning,
        VisualBertModel,
    )


class VisualBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        visual_seq_length=5,
        is_training=True,
        use_attention_mask=True,
        use_visual_attention_mask=True,
        use_token_type_ids=True,
        use_visual_token_type_ids=True,
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
        visual_embedding_dim=20,
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
        self.visual_seq_length = visual_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_visual_attention_mask = use_visual_attention_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_visual_token_type_ids = use_visual_token_type_ids
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
        self.visual_embedding_dim = visual_embedding_dim
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def get_config(self):
        return VisualBertConfig(
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
            visual_embedding_dim=self.visual_embedding_dim,
            num_labels=self.num_labels,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        visual_embeds = floats_tensor([self.batch_size, self.visual_seq_length, self.visual_embedding_dim])

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)

        visual_attention_mask = None
        if self.use_visual_attention_mask:
            visual_attention_mask = torch.ones(
                (self.batch_size, self.visual_seq_length), dtype=torch.long, device=torch_device
            )

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        visual_token_type_ids = None
        if self.use_visual_token_type_ids:
            visual_token_type_ids = ids_tensor([self.batch_size, self.visual_seq_length], self.type_vocab_size)

        config = self.get_config()
        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }

    def prepare_config_and_inputs_for_pretraining(self):
        masked_lm_labels = None
        sentence_image_labels = None

        if self.use_labels:
            masked_lm_labels = ids_tensor([self.batch_size, self.seq_length + self.visual_seq_length], self.vocab_size)
            sentence_image_labels = ids_tensor(
                [self.batch_size],
                self.type_sequence_label_size,
            )

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": masked_lm_labels, "sentence_image_labels": sentence_image_labels})

        return config, input_dict

    def prepare_config_and_inputs_for_multiple_choice(self):
        input_ids = ids_tensor([self.batch_size, self.num_choices, self.seq_length], self.vocab_size)
        visual_embeds = floats_tensor(
            [self.batch_size, self.num_choices, self.visual_seq_length, self.visual_embedding_dim]
        )

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = torch.ones(
                (self.batch_size, self.num_choices, self.seq_length), dtype=torch.long, device=torch_device
            )

        visual_attention_mask = None
        if self.use_visual_attention_mask:
            visual_attention_mask = torch.ones(
                (self.batch_size, self.num_choices, self.visual_seq_length), dtype=torch.long, device=torch_device
            )

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.num_choices, self.seq_length], self.type_vocab_size)

        visual_token_type_ids = None
        if self.use_visual_token_type_ids:
            visual_token_type_ids = ids_tensor(
                [self.batch_size, self.num_choices, self.visual_seq_length], self.type_vocab_size
            )

        labels = None

        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "labels": labels,
        }

    def prepare_config_and_inputs_for_vqa(self):
        vqa_labels = None

        if self.use_labels:
            vqa_labels = floats_tensor([self.batch_size, self.num_labels])

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": vqa_labels})
        return config, input_dict

    def prepare_config_and_inputs_for_nlvr(self):
        nlvr_labels = None

        if self.use_labels:
            nlvr_labels = ids_tensor([self.batch_size], self.num_labels)

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": nlvr_labels})
        return config, input_dict

    def prepare_config_and_inputs_for_flickr(self):
        region_to_phrase_position = torch.cat(
            (
                ids_tensor([self.batch_size, self.seq_length], self.visual_seq_length),
                torch.ones(self.batch_size, self.visual_seq_length, dtype=torch.long, device=torch_device) * -1,
            ),
            dim=-1,
        )
        flickr_labels = None
        if self.use_labels:
            flickr_labels = floats_tensor(
                [self.batch_size, self.seq_length + self.visual_seq_length, self.visual_seq_length]
            )

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"region_to_phrase_position": region_to_phrase_position, "labels": flickr_labels})
        return config, input_dict

    def create_and_check_model(self, config, input_dict):
        model = VisualBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length + self.visual_seq_length, self.hidden_size),
        )

    def create_and_check_for_pretraining(self, config, input_dict):
        model = VisualBertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(
            result.prediction_logits.shape,
            (self.batch_size, self.seq_length + self.visual_seq_length, self.vocab_size),
        )

    def create_and_check_for_vqa(self, config, input_dict):
        model = VisualBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(self, config, input_dict):
        model = VisualBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_nlvr(self, config, input_dict):
        model = VisualBertForVisualReasoning(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_flickr(self, config, input_dict):
        model = VisualBertForRegionToPhraseAlignment(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.seq_length + self.visual_seq_length, self.visual_seq_length)
        )


@require_torch
class VisualBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            VisualBertModel,
            VisualBertForMultipleChoice,
            VisualBertForVisualReasoning,
            VisualBertForRegionToPhraseAlignment,
            VisualBertForQuestionAnswering,
            VisualBertForPreTraining,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"feature-extraction": VisualBertModel} if is_torch_available() else {}
    test_torchscript = False
    test_pruning = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class == VisualBertForMultipleChoice:
            for key in inputs_dict.keys():
                value = inputs_dict[key]
                if isinstance(value, torch.Tensor) and value.ndim > 1:
                    if key != "visual_embeds":
                        inputs_dict[key] = (
                            inputs_dict[key].unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                        )
                    else:
                        inputs_dict[key] = (
                            inputs_dict[key]
                            .unsqueeze(1)
                            .expand(-1, self.model_tester.num_choices, -1, self.model_tester.visual_embedding_dim)
                            .contiguous()
                        )

        elif model_class == VisualBertForRegionToPhraseAlignment:
            total_length = self.model_tester.seq_length + self.model_tester.visual_seq_length
            batch_size = self.model_tester.batch_size
            inputs_dict["region_to_phrase_position"] = torch.zeros(
                (batch_size, total_length),
                dtype=torch.long,
                device=torch_device,
            )

        if return_labels:
            if model_class == VisualBertForMultipleChoice:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class == VisualBertForPreTraining:
                total_length = self.model_tester.seq_length + self.model_tester.visual_seq_length
                batch_size = self.model_tester.batch_size
                inputs_dict["labels"] = torch.zeros(
                    (batch_size, total_length),
                    dtype=torch.long,
                    device=torch_device,
                )
                inputs_dict["sentence_image_labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

            # Flickr expects float labels
            elif model_class == VisualBertForRegionToPhraseAlignment:
                batch_size = self.model_tester.batch_size
                total_length = self.model_tester.seq_length + self.model_tester.visual_seq_length

                inputs_dict["labels"] = torch.ones(
                    (
                        batch_size,
                        total_length,
                        self.model_tester.visual_seq_length,
                    ),
                    dtype=torch.float,
                    device=torch_device,
                )

            # VQA expects float labels
            elif model_class == VisualBertForQuestionAnswering:
                inputs_dict["labels"] = torch.ones(
                    (self.model_tester.batch_size, self.model_tester.num_labels),
                    dtype=torch.float,
                    device=torch_device,
                )

            elif model_class == VisualBertForVisualReasoning:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size), dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = VisualBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VisualBertConfig, hidden_size=37)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        visual_seq_len = getattr(self.model_tester, "visual_seq_length", None)

        encoder_seq_length = (seq_len if seq_len is not None else 0) + (
            visual_seq_len if visual_seq_len is not None else 0
        )
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
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

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length + self.model_tester.visual_seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pretraining()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_model_for_vqa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_vqa()
        self.model_tester.create_and_check_for_vqa(*config_and_inputs)

    def test_model_for_nlvr(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_nlvr()
        self.model_tester.create_and_check_for_nlvr(*config_and_inputs)

    def test_model_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_multiple_choice()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_model_for_flickr(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_flickr()
        self.model_tester.create_and_check_for_flickr(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "uclanlp/visualbert-vqa"
        model = VisualBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

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
class VisualBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_vqa_coco_pre(self):
        model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.long)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-5.1858, -5.1903, -4.9142], [-6.2214, -5.9238, -5.8381], [-6.3027, -5.9939, -5.9297]]]
        )

        torch.testing.assert_close(output.prediction_logits[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor([[0.7393, 0.1754]])

        torch.testing.assert_close(output.seq_relationship_logits, expected_slice_2, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_vqa(self):
        model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.long)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        # vocab_size = 30522

        expected_shape = torch.Size((1, 3129))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-8.9898, 3.0803, -1.8016, 2.4542, -8.3420, -2.0224, -3.3124, -4.4139, -3.1491, -3.8997]]
        )

        torch.testing.assert_close(output.logits[:, :10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_nlvr(self):
        model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 1024), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.long)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        # vocab_size = 30522

        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor([[-1.1436, 0.8900]])

        torch.testing.assert_close(output.logits, expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_vcr(self):
        model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

        input_ids = torch.tensor([[[1, 2, 3, 4, 5, 6] for i in range(4)]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.ones_like(input_ids)

        visual_embeds = torch.ones(size=(1, 4, 10, 512), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 4, 10), dtype=torch.long)
        visual_attention_mask = torch.ones_like(visual_token_type_ids)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        # vocab_size = 30522

        expected_shape = torch.Size((1, 4))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor([[-7.7697, -7.7697, -7.7697, -7.7697]])

        torch.testing.assert_close(output.logits, expected_slice, rtol=1e-4, atol=1e-4)
