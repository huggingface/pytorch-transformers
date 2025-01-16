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
"""Testing suite for the PyTorch MobileNetV1 model."""

import unittest

from transformers import MobileNetV1Config
from transformers.testing_utils import is_flaky, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import MobileNetV1ForImageClassification, MobileNetV1Model


if is_vision_available():
    from PIL import Image

    from transformers import MobileNetV1ImageProcessor


class MobileNetV1ConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "tf_padding"))
        self.parent.assertTrue(hasattr(config, "depth_multiplier"))


class MobileNetV1ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        image_size=32,
        depth_multiplier=0.25,
        min_depth=8,
        tf_padding=True,
        last_hidden_size=1024,
        output_stride=32,
        hidden_act="relu6",
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
        self.min_depth = min_depth
        self.tf_padding = tf_padding
        self.last_hidden_size = int(last_hidden_size * depth_multiplier)
        self.output_stride = output_stride
        self.hidden_act = hidden_act
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
        return MobileNetV1Config(
            num_channels=self.num_channels,
            image_size=self.image_size,
            depth_multiplier=self.depth_multiplier,
            min_depth=self.min_depth,
            tf_padding=self.tf_padding,
            hidden_act=self.hidden_act,
            classifier_dropout_prob=self.classifier_dropout_prob,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = MobileNetV1Model(config=config)
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

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = MobileNetV1ForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class MobileNetV1ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MobileNetV1 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (MobileNetV1Model, MobileNetV1ForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": MobileNetV1Model, "image-classification": MobileNetV1ForImageClassification}
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = MobileNetV1ModelTester(self)
        self.config_tester = MobileNetV1ConfigTester(self, config_class=MobileNetV1Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MobileNetV1 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MobileNetV1 does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="MobileNetV1 does not output attentions")
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

            expected_num_stages = 26
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

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/mobilenet_v1_1.0_224"
        model = MobileNetV1Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    # TODO: ydshieh
    @unittest.skip("skip for now as #35564 fails this test more frequently for this model")
    @is_flaky(description="is_flaky https://github.com/huggingface/transformers/pull/31258")
    def test_batching_equivalence(self):
        super().test_batching_equivalence()


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class MobileNetV1ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            MobileNetV1ImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = MobileNetV1ForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1001))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-4.1739, -1.1233, 3.1205]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)
