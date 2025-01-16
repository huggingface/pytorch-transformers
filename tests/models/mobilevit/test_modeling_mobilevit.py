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
"""Testing suite for the PyTorch MobileViT model."""

import unittest

from transformers import MobileViTConfig
from transformers.testing_utils import is_flaky, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import MobileViTForImageClassification, MobileViTForSemanticSegmentation, MobileViTModel


if is_vision_available():
    from PIL import Image

    from transformers import MobileViTImageProcessor


class MobileViTConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "neck_hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))


class MobileViTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=2,
        num_channels=3,
        last_hidden_size=32,
        num_attention_heads=4,
        hidden_act="silu",
        conv_kernel_size=3,
        output_stride=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.last_hidden_size = last_hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
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
        return MobileViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            conv_kernel_size=self.conv_kernel_size,
            output_stride=self.output_stride,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            classifier_dropout_prob=self.classifier_dropout_prob,
            initializer_range=self.initializer_range,
            hidden_sizes=[12, 16, 20],
            neck_hidden_sizes=[8, 8, 16, 16, 32, 32, 32],
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = MobileViTModel(config=config)
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
        model = MobileViTForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = MobileViTForSemanticSegmentation(config)
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
class MobileViTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MobileViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": MobileViTModel,
            "image-classification": MobileViTForImageClassification,
            "image-segmentation": MobileViTForSemanticSegmentation,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = MobileViTModelTester(self)
        self.config_tester = MobileViTConfigTester(self, config_class=MobileViTConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MobileViT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MobileViT does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="MobileViT does not output attentions")
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

            expected_num_stages = 5
            self.assertEqual(len(hidden_states), expected_num_stages)

            # MobileViT's feature maps are of shape (batch_size, num_channels, height, width)
            # with the width and height being successively divided by 2.
            divisor = 2
            for i in range(len(hidden_states)):
                self.assertListEqual(
                    list(hidden_states[i].shape[-2:]),
                    [self.model_tester.image_size // divisor, self.model_tester.image_size // divisor],
                )
                divisor *= 2

            self.assertEqual(self.model_tester.output_stride, divisor // 2)

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
        model_name = "apple/mobilevit-small"
        model = MobileViTModel.from_pretrained(model_name)
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
class MobileViTModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return MobileViTImageProcessor.from_pretrained("apple/mobilevit-xx-small") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.9364, -1.2327, -0.4653]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_semantic_segmentation(self):
        model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")
        model = model.to(torch_device)

        image_processor = MobileViTImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 21, 32, 32))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [[6.9713, 6.9786, 7.2422], [7.2893, 7.2825, 7.4446], [7.6580, 7.8797, 7.9420]],
                [[-10.6869, -10.3250, -10.3471], [-10.4228, -9.9868, -9.7132], [-11.0405, -11.0221, -10.7318]],
                [[-3.3089, -2.8539, -2.6740], [-3.2706, -2.5621, -2.5108], [-3.2534, -2.6615, -2.6651]],
            ],
            device=torch_device,
        )

        torch.testing.assert_close(logits[0, :3, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_post_processing_semantic_segmentation(self):
        model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")
        model = model.to(torch_device)

        image_processor = MobileViTImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        outputs.logits = outputs.logits.detach().cpu()

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[(50, 60)])
        expected_shape = torch.Size((50, 60))
        self.assertEqual(segmentation[0].shape, expected_shape)

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = torch.Size((32, 32))
        self.assertEqual(segmentation[0].shape, expected_shape)
