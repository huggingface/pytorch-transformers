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
"""Testing suite for the PyTorch Dinat model."""

import collections
import unittest

from transformers import DinatConfig
from transformers.testing_utils import require_natten, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DinatBackbone, DinatForImageClassification, DinatModel

if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class DinatModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=64,
        patch_size=4,
        num_channels=3,
        embed_dim=16,
        depths=[1, 2, 1],
        num_heads=[2, 4, 8],
        kernel_size=3,
        dilations=[[3], [1, 2], [1]],
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        patch_norm=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        scope=None,
        use_labels=True,
        num_labels=10,
        out_features=["stage1", "stage2"],
        out_indices=[1, 2],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.patch_norm = patch_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.scope = scope
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.out_features = out_features
        self.out_indices = out_indices

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DinatConfig(
            num_labels=self.num_labels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            drop_path_rate=self.drop_path_rate,
            hidden_act=self.hidden_act,
            patch_norm=self.patch_norm,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            out_features=self.out_features,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DinatModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_height = expected_width = (config.image_size // config.patch_size) // (2 ** (len(config.depths) - 1))
        expected_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, expected_height, expected_width, expected_dim)
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        model = DinatForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        # test greyscale images
        config.num_channels = 1
        model = DinatForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = DinatBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], 16, 16])

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))

        # verify backbone works with out_features=None
        config.out_features = None
        model = DinatBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, model.channels[-1], 4, 4])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_natten
@require_torch
class DinatModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DinatModel,
            DinatForImageClassification,
            DinatBackbone,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-feature-extraction": DinatModel, "image-classification": DinatForImageClassification}
        if is_torch_available()
        else {}
    )
    fx_compatible = False

    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DinatModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=DinatConfig, embed_dim=37, common_properties=["patch_size", "num_channels"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    @unittest.skip(reason="Dinat does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Dinat does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_attention_outputs(self):
        self.skipTest(reason="Dinat's attention operation is handled entirely by NATTEN.")

    def check_hidden_states_output(self, inputs_dict, config, model_class, image_size):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

        hidden_states = outputs.hidden_states

        expected_num_layers = getattr(
            self.model_tester, "expected_num_hidden_layers", len(self.model_tester.depths) + 1
        )
        self.assertEqual(len(hidden_states), expected_num_layers)

        # Dinat has a different seq_length
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )

        height = image_size[0] // patch_size[0]
        width = image_size[1] // patch_size[1]

        self.assertListEqual(
            list(hidden_states[0].shape[-3:]),
            [height, width, self.model_tester.embed_dim],
        )

        if model_class.__name__ != "DinatBackbone":
            reshaped_hidden_states = outputs.reshaped_hidden_states
            self.assertEqual(len(reshaped_hidden_states), expected_num_layers)

            batch_size, num_channels, height, width = reshaped_hidden_states[0].shape
            reshaped_hidden_states = (
                reshaped_hidden_states[0].view(batch_size, num_channels, height, width).permute(0, 2, 3, 1)
            )
            self.assertListEqual(
                list(reshaped_hidden_states.shape[-3:]),
                [height, width, self.model_tester.embed_dim],
            )

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

    @slow
    def test_model_from_pretrained(self):
        model_name = "shi-labs/dinat-mini-in1k-224"
        model = DinatModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "embeddings" not in name and param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@require_natten
@require_vision
@require_torch
class DinatModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = DinatForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224").to(torch_device)
        image_processor = self.default_image_processor

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.1545, -0.7667, 0.4642]).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)


@require_torch
@require_natten
class DinatBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (DinatBackbone,) if is_torch_available() else ()
    config_class = DinatConfig

    def setUp(self):
        self.model_tester = DinatModelTester(self)
