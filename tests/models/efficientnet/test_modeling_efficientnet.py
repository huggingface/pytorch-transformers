# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EfficientNet model."""

import unittest

from transformers import EfficientNetConfig
from transformers.testing_utils import is_pipeline_test, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import EfficientNetForImageClassification, EfficientNetModel


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class EfficientNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        num_channels=3,
        kernel_sizes=[3, 3, 5],
        in_channels=[32, 16, 24],
        out_channels=[16, 24, 20],
        strides=[1, 1, 2],
        num_block_repeats=[1, 1, 2],
        expand_ratios=[1, 6, 6],
        is_training=True,
        use_labels=True,
        intermediate_size=37,
        hidden_act="gelu",
        num_labels=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.is_training = is_training
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return EfficientNetConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            kernel_sizes=self.kernel_sizes,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=self.strides,
            num_block_repeats=self.num_block_repeats,
            expand_ratios=self.expand_ratios,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = EfficientNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected last hidden states: B, C, H // 4, W // 4
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, config.hidden_dim, self.image_size // 4, self.image_size // 4),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        model = EfficientNetForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class EfficientNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as EfficientNet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (EfficientNetModel, EfficientNetForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": EfficientNetModel, "image-classification": EfficientNetForImageClassification}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = EfficientNetModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=EfficientNetConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["num_channels", "image_size", "hidden_dim"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="EfficientNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EfficientNet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="EfficientNet does not use feedforward chunking")
    def test_feed_forward_chunking(self):
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

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            num_blocks = sum(config.num_block_repeats) * 4
            self.assertEqual(len(hidden_states), num_blocks)

            # EfficientNet's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 2, self.model_tester.image_size // 2],
            )

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
        model_name = "google/efficientnet-b7"
        model = EfficientNetModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @is_pipeline_test
    @require_vision
    @slow
    def test_pipeline_image_feature_extraction(self):
        super().test_pipeline_image_feature_extraction()

    @is_pipeline_test
    @require_vision
    @slow
    def test_pipeline_image_feature_extraction_fp16(self):
        super().test_pipeline_image_feature_extraction_fp16()

    @is_pipeline_test
    @require_vision
    @slow
    def test_pipeline_image_classification(self):
        super().test_pipeline_image_classification()


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class EfficientNetModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("google/efficientnet-b7") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.2962, 0.4487, 0.4499]).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)
