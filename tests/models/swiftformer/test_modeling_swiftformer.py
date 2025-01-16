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
"""Testing suite for the PyTorch SwiftFormer model."""

import copy
import unittest

from transformers import PretrainedConfig, SwiftFormerConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import SwiftFormerForImageClassification, SwiftFormerModel


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor


class SwiftFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        image_size=224,
        num_labels=3,
        layer_depths=[1, 1, 1, 1],
        embed_dims=[16, 16, 32, 32],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_labels = num_labels
        self.image_size = image_size
        self.layer_depths = layer_depths
        self.embed_dims = embed_dims

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return SwiftFormerConfig(
            depths=self.layer_depths,
            embed_dims=self.embed_dims,
            mlp_ratio=4,
            downsamples=[True, True, True, True],
            hidden_act="gelu",
            num_labels=self.num_labels,
            down_patch_size=3,
            down_stride=2,
            down_pad=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = SwiftFormerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.embed_dims[-1], 7, 7))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = SwiftFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        model = SwiftFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        (config, pixel_values, labels) = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SwiftFormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SwiftFormer does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SwiftFormerModel, SwiftFormerForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": SwiftFormerModel, "image-classification": SwiftFormerForImageClassification}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = SwiftFormerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=SwiftFormerConfig,
            has_text_modality=False,
            hidden_size=37,
            num_attention_heads=12,
            num_hidden_layers=12,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SwiftFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "MBZUAI/swiftformer-xs"
        model = SwiftFormerModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="SwiftFormer does not output attentions")
    def test_attention_outputs(self):
        pass

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = 8
            self.assertEqual(len(hidden_states), expected_num_stages)  # TODO

            # SwiftFormer's feature maps are of shape (batch_size, embed_dims, height, width)
            # with the width and height being successively divided by 2, after every 2 blocks
            for i in range(len(hidden_states)):
                self.assertEqual(
                    hidden_states[i].shape,
                    torch.Size(
                        [
                            self.model_tester.batch_size,
                            self.model_tester.embed_dims[i // 2],
                            (self.model_tester.image_size // 4) // 2 ** (i // 2),
                            (self.model_tester.image_size // 4) // 2 ** (i // 2),
                        ]
                    ),
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        def _config_zero_init(config):
            configs_no_init = copy.deepcopy(config)
            for key in configs_no_init.__dict__.keys():
                if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
                    setattr(configs_no_init, key, 1e-10)
                if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
                    no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
                    setattr(configs_no_init, key, no_init_subconfig)
            return configs_no_init

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9) / 1e9).round().item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class SwiftFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return ViTImageProcessor.from_pretrained("MBZUAI/swiftformer-xs") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = SwiftFormerForImageClassification.from_pretrained("MBZUAI/swiftformer-xs").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([[-2.1703e00, 2.1107e00, -2.0811e00]]).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)
