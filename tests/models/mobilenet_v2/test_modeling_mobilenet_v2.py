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
"""Testing suite for the PyTorch MobileNetV2 model."""

import unittest

from transformers import MobileNetV2Config
from transformers.testing_utils import is_flaky, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import MobileNetV2ForImageClassification, MobileNetV2ForSemanticSegmentation, MobileNetV2Model


if is_vision_available():
    from PIL import Image

    from transformers import MobileNetV2ImageProcessor


class MobileNetV2ConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "tf_padding"))
        self.parent.assertTrue(hasattr(config, "depth_multiplier"))


class MobileNetV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        image_size=32,
        depth_multiplier=0.25,
        depth_divisible_by=8,
        min_depth=8,
        expand_ratio=6,
        output_stride=32,
        first_layer_is_expansion=True,
        finegrained_output=True,
        tf_padding=True,
        hidden_act="relu6",
        last_hidden_size=1280,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        self.depth_divisible_by = depth_divisible_by
        self.min_depth = min_depth
        self.expand_ratio = expand_ratio
        self.tf_padding = tf_padding
        self.output_stride = output_stride
        self.first_layer_is_expansion = first_layer_is_expansion
        self.finegrained_output = finegrained_output
        self.hidden_act = hidden_act
        self.last_hidden_size = last_hidden_size if finegrained_output else int(last_hidden_size * depth_multiplier)
        self.classifier_dropout_prob = classifier_dropout_prob
        self.use_labels = use_labels
        self.is_training = is_training
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels, pixel_labels

    def get_config(self):
        return MobileNetV2Config(
            num_channels=self.num_channels,
            image_size=self.image_size,
            depth_multiplier=self.depth_multiplier,
            depth_divisible_by=self.depth_divisible_by,
            min_depth=self.min_depth,
            expand_ratio=self.expand_ratio,
            output_stride=self.output_stride,
            first_layer_is_expansion=self.first_layer_is_expansion,
            finegrained_output=self.finegrained_output,
            hidden_act=self.hidden_act,
            tf_padding=self.tf_padding,
            classifier_dropout_prob=self.classifier_dropout_prob,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = MobileNetV2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (
                self.batch_size,
                self.last_hidden_size,
                self.image_size // self.output_stride,
                self.image_size // self.output_stride,
            ),
        )
        self.parent.assertEqual(
            result.pooler_output.shape,
            (self.batch_size, self.last_hidden_size),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = MobileNetV2ForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = MobileNetV2ForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape,
            (
                self.batch_size,
                self.num_labels,
                self.image_size // self.output_stride,
                self.image_size // self.output_stride,
            ),
        )
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(
            result.logits.shape,
            (
                self.batch_size,
                self.num_labels,
                self.image_size // self.output_stride,
                self.image_size // self.output_stride,
            ),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class MobileNetV2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MobileNetV2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (MobileNetV2Model, MobileNetV2ForImageClassification, MobileNetV2ForSemanticSegmentation)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": MobileNetV2Model,
            "image-classification": MobileNetV2ForImageClassification,
            "image-segmentation": MobileNetV2ForSemanticSegmentation,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = MobileNetV2ModelTester(self)
        self.config_tester = MobileNetV2ConfigTester(self, config_class=MobileNetV2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MobileNetV2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MobileNetV2 does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="MobileNetV2 does not output attentions")
    def test_attention_outputs(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = 16
            self.assertEqual(len(hidden_states), expected_num_stages)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/mobilenet_v2_1.4_224"
        model = MobileNetV2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @is_flaky(description="is_flaky https://github.com/huggingface/transformers/issues/29516")
    def test_batching_equivalence(self):
        super().test_batching_equivalence()


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class MobileNetV2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            MobileNetV2ImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1001))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([0.2445, -1.1993, 0.1905]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_semantic_segmentation(self):
        model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
        model = model.to(torch_device)

        image_processor = MobileNetV2ImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 21, 65, 65))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [[17.5790, 17.7581, 18.3355], [18.3257, 18.4230, 18.8973], [18.6169, 18.8650, 19.2187]],
                [[-2.1595, -2.0977, -2.3741], [-2.4226, -2.3028, -2.6835], [-2.7819, -2.5991, -2.7706]],
                [[4.2058, 4.8317, 4.7638], [4.4136, 5.0361, 4.9383], [4.5028, 4.9644, 4.8734]],
            ],
            device=torch_device,
        )

        torch.testing.assert_close(logits[0, :3, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
