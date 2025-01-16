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
"""Testing suite for the PyTorch LayoutLMv3 model."""

import copy
import unittest

from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        LayoutLMv3Config,
        LayoutLMv3ForQuestionAnswering,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Model,
    )

if is_vision_available():
    from PIL import Image

    from transformers import LayoutLMv3ImageProcessor


class LayoutLMv3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=4,
        patch_size=2,
        text_seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=36,
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
        coordinate_size=6,
        shape_size=6,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.text_seq_length = text_seq_length
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
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.range_bbox = range_bbox

        # LayoutLMv3's sequence length equals the number of text tokens + number of patches + 1 (we add 1 for the CLS token)
        self.text_seq_length = text_seq_length
        self.image_seq_length = (image_size // patch_size) ** 2 + 1
        self.seq_length = self.text_seq_length + self.image_seq_length

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.vocab_size)

        bbox = ids_tensor([self.batch_size, self.text_seq_length, 4], self.range_bbox)
        # Ensure that bbox is legal
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                if bbox[i, j, 3] < bbox[i, j, 1]:
                    t = bbox[i, j, 3]
                    bbox[i, j, 3] = bbox[i, j, 1]
                    bbox[i, j, 1] = t
                if bbox[i, j, 2] < bbox[i, j, 0]:
                    t = bbox[i, j, 2]
                    bbox[i, j, 2] = bbox[i, j, 0]
                    bbox[i, j, 0] = t

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.text_seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.text_seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.text_seq_length], self.num_labels)

        config = LayoutLMv3Config(
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
            coordinate_size=self.coordinate_size,
            shape_size=self.shape_size,
            input_size=self.image_size,
            patch_size=self.patch_size,
        )

        return config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels

    def create_and_check_model(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = LayoutLMv3Model(config=config)
        model.to(torch_device)
        model.eval()

        # text + image
        result = model(input_ids, pixel_values=pixel_values)
        result = model(
            input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=input_mask, token_type_ids=token_type_ids
        )
        result = model(input_ids, bbox=bbox, pixel_values=pixel_values, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, pixel_values=pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

        # text only
        result = model(input_ids)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.text_seq_length, self.hidden_size)
        )

        # image only
        result = model(pixel_values=pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.image_seq_length, self.hidden_size)
        )

    def create_and_check_for_sequence_classification(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv3ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv3ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.text_seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = LayoutLMv3ForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class LayoutLMv3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_pruning = False
    test_torchscript = False
    test_mismatched_shapes = False

    all_model_classes = (
        (
            LayoutLMv3Model,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3ForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"document-question-answering": LayoutLMv3ForQuestionAnswering, "feature-extraction": LayoutLMv3Model}
        if is_torch_available()
        else {}
    )

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
        # `DocumentQuestionAnsweringPipeline` is expected to work with this model, but it combines the text and visual
        # embedding along the sequence dimension (dim 1), which causes an error during post-processing as `p_mask` has
        # the sequence dimension of the text embedding only.
        # (see the line `embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)`)
        return True

    def setUp(self):
        self.model_tester = LayoutLMv3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMv3Config, hidden_size=37)

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
            elif model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.text_seq_length),
                    dtype=torch.long,
                    device=torch_device,
                )

        return inputs_dict

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

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "microsoft/layoutlmv3-base"
        model = LayoutLMv3Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
class LayoutLMv3ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return LayoutLMv3ImageProcessor(apply_ocr=False) if is_vision_available() else None

    @slow
    def test_inference_no_head(self):
        model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        input_ids = torch.tensor([[1, 2]])
        bbox = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).unsqueeze(0)

        # forward pass
        outputs = model(
            input_ids=input_ids.to(torch_device),
            bbox=bbox.to(torch_device),
            pixel_values=pixel_values.to(torch_device),
        )

        # verify the logits
        expected_shape = torch.Size((1, 199, 768))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.0529, 0.3618, 0.1632], [-0.1587, -0.1667, -0.0400], [-0.1557, -0.1671, -0.0505]]
        ).to(torch_device)

        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
