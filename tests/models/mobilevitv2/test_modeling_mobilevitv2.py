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
"""Testing suite for the PyTorch MobileViTV2 model."""

import unittest

from transformers import MobileViTV2Config
from transformers.testing_utils import require_torch, require_torch_multi_gpu, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import MobileViTV2ForImageClassification, MobileViTV2ForSemanticSegmentation, MobileViTV2Model
    from transformers.models.mobilevitv2.modeling_mobilevitv2 import (
        make_divisible,
    )


if is_vision_available():
    from PIL import Image

    from transformers import MobileViTImageProcessor


class MobileViTV2ConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "width_multiplier"))


class MobileViTV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=64,
        patch_size=2,
        num_channels=3,
        hidden_act="swish",
        conv_kernel_size=3,
        output_stride=32,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=10,
        scope=None,
        width_multiplier=0.25,
        ffn_dropout=0.0,
        attn_dropout=0.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.last_hidden_size = make_divisible(512 * width_multiplier, divisor=8)
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.classifier_dropout_prob = classifier_dropout_prob
        self.use_labels = use_labels
        self.is_training = is_training
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.scope = scope
        self.width_multiplier = width_multiplier
        self.ffn_dropout_prob = ffn_dropout
        self.attn_dropout_prob = attn_dropout

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
        return MobileViTV2Config(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_act=self.hidden_act,
            conv_kernel_size=self.conv_kernel_size,
            output_stride=self.output_stride,
            classifier_dropout_prob=self.classifier_dropout_prob,
            initializer_range=self.initializer_range,
            width_multiplier=self.width_multiplier,
            ffn_dropout=self.ffn_dropout_prob,
            attn_dropout=self.attn_dropout_prob,
            base_attn_unit_dims=[16, 24, 32],
            n_attn_blocks=[1, 1, 2],
            aspp_out_channels=32,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = MobileViTV2Model(config=config)
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
        model = MobileViTV2ForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = MobileViTV2ForSemanticSegmentation(config)
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
class MobileViTV2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MobileViTV2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (MobileViTV2Model, MobileViTV2ForImageClassification, MobileViTV2ForSemanticSegmentation)
        if is_torch_available()
        else ()
    )

    pipeline_model_mapping = (
        {
            "image-feature-extraction": MobileViTV2Model,
            "image-classification": MobileViTV2ForImageClassification,
            "image-segmentation": MobileViTV2ForSemanticSegmentation,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = MobileViTV2ModelTester(self)
        self.config_tester = MobileViTV2ConfigTester(self, config_class=MobileViTV2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MobileViTV2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MobileViTV2 does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="MobileViTV2 does not output attentions")
    def test_attention_outputs(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(reason="Got `CUDA error: misaligned address` for tests after this one being run.")
    def test_multi_gpu_data_parallel_forward(self):
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

            # MobileViTV2's feature maps are of shape (batch_size, num_channels, height, width)
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
        model_name = "apple/mobilevitv2-1.0-imagenet1k-256"
        model = MobileViTV2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class MobileViTV2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            MobileViTImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(
            torch_device
        )

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.6336e00, -7.3204e-02, -5.1883e-01]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_semantic_segmentation(self):
        model = MobileViTV2ForSemanticSegmentation.from_pretrained("shehan97/mobilevitv2-1.0-voc-deeplabv3")
        model = model.to(torch_device)

        image_processor = MobileViTImageProcessor.from_pretrained("shehan97/mobilevitv2-1.0-voc-deeplabv3")

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
                [[7.0863, 7.1525, 6.8201], [6.6931, 6.8770, 6.8933], [6.2978, 7.0366, 6.9636]],
                [[-3.7134, -3.6712, -3.6675], [-3.5825, -3.3549, -3.4777], [-3.3435, -3.3979, -3.2857]],
                [[-2.9329, -2.8003, -2.7369], [-3.0564, -2.4780, -2.0207], [-2.6889, -1.9298, -1.7640]],
            ],
            device=torch_device,
        )

        torch.testing.assert_close(logits[0, :3, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_post_processing_semantic_segmentation(self):
        model = MobileViTV2ForSemanticSegmentation.from_pretrained("shehan97/mobilevitv2-1.0-voc-deeplabv3")
        model = model.to(torch_device)

        image_processor = MobileViTImageProcessor.from_pretrained("shehan97/mobilevitv2-1.0-voc-deeplabv3")

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
