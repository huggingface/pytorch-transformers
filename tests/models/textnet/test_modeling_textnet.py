# coding=utf-8
# Copyright 2024 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch TextNet model."""

import unittest

import requests
from PIL import Image

from transformers import TextNetConfig
from transformers.models.textnet.image_processing_textnet import TextNetImageProcessor
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import TextNetBackbone, TextNetForImageClassification, TextNetModel


class TextNetConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        self.parent.assertTrue(hasattr(config, "num_encoder_blocks"))


class TextNetModelTester:
    def __init__(
        self,
        parent,
        stem_kernel_size=3,
        stem_stride=2,
        stem_in_channels=3,
        stem_out_channels=32,
        stem_act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
        conv_layer_kernel_sizes=[
            [[3, 3]],
            [[3, 3]],
            [[3, 3]],
            [[3, 3]],
        ],
        conv_layer_strides=[
            [2],
            [2],
            [2],
            [2],
        ],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        out_indices=[1, 2, 3, 4],
        batch_size=3,
        num_channels=3,
        image_size=[32, 32],
        is_training=True,
        use_labels=True,
        num_labels=3,
        hidden_sizes=[32, 32, 32, 32, 32],
    ):
        self.parent = parent
        self.stem_kernel_size = stem_kernel_size
        self.stem_stride = stem_stride
        self.stem_in_channels = stem_in_channels
        self.stem_out_channels = stem_out_channels
        self.act_func = stem_act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order
        self.conv_layer_kernel_sizes = conv_layer_kernel_sizes
        self.conv_layer_strides = conv_layer_strides

        self.out_features = out_features
        self.out_indices = out_indices

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.hidden_sizes = hidden_sizes

        self.num_stages = 5

    def get_config(self):
        return TextNetConfig(
            stem_kernel_size=self.stem_kernel_size,
            stem_stride=self.stem_stride,
            stem_num_channels=self.stem_in_channels,
            stem_out_channels=self.stem_out_channels,
            act_func=self.act_func,
            dropout_rate=self.dropout_rate,
            ops_order=self.ops_order,
            conv_layer_kernel_sizes=self.conv_layer_kernel_sizes,
            conv_layer_strides=self.conv_layer_strides,
            out_features=self.out_features,
            out_indices=self.out_indices,
            hidden_sizes=self.hidden_sizes,
            image_size=self.image_size,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TextNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        scale_h = self.image_size[0] // 32
        scale_w = self.image_size[1] // 32
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_sizes[-1], scale_h, scale_w),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = TextNetForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = TextNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        scale_h = self.image_size[0] // 32
        scale_w = self.image_size[1] // 32
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 8 * scale_h, 8 * scale_w]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])

        # verify backbone works with out_features=None
        config.out_features = None
        model = TextNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        scale_h = self.image_size[0] // 32
        scale_w = self.image_size[1] // 32
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[0], scale_h, scale_w]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class TextNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some tests of test_modeling_common.py, as TextNet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (TextNetModel, TextNetForImageClassification, TextNetBackbone) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": TextNetModel, "image-classification": TextNetForImageClassification}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = TextNetModelTester(self)
        self.config_tester = TextNetConfigTester(self, config_class=TextNetConfig, has_text_modality=False)

    @unittest.skip(reason="TextNet does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="TextNet does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="TextNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TextNet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.assertTrue(
                        torch.all(module.weight == 1),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                    self.assertTrue(
                        torch.all(module.bias == 0),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            self.assertEqual(len(hidden_states), self.model_tester.num_stages)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size[0] // 2, self.model_tester.image_size[1] // 2],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        layers_type = ["preactivation", "bottleneck"]
        for model_class in self.all_model_classes:
            for layer_type in layers_type:
                config.layer_type = layer_type
                inputs_dict["output_hidden_states"] = True
                check_hidden_states_output(inputs_dict, config, model_class)

                # check that output_hidden_states also work using config
                del inputs_dict["output_hidden_states"]
                config.output_hidden_states = True

                check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="TextNet does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "czczup/textnet-base"
        model = TextNetModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
@require_vision
class TextNetModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        processor = TextNetImageProcessor.from_pretrained("czczup/textnet-base")
        model = TextNetModel.from_pretrained("czczup/textnet-base").to(torch_device)

        # prepare image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            output = model(**inputs)

        # verify logits
        self.assertEqual(output.logits.shape, torch.Size([1, 2]))
        expected_slice_backbone = torch.tensor(
            [0.9210, 0.6099, 0.0000, 0.0000, 0.0000, 0.0000, 3.2207, 2.6602, 1.8925, 0.0000],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(output.feature_maps[-1][0][10][12][:10], expected_slice_backbone, atol=1e-3))


@require_torch
# Copied from tests.models.bit.test_modeling_bit.BitBackboneTest with Bit->TextNet
class TextNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (TextNetBackbone,) if is_torch_available() else ()
    config_class = TextNetConfig

    has_attentions = False

    def setUp(self):
        self.model_tester = TextNetModelTester(self)
