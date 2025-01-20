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
"""Testing suite for the PyTorch ViTDet model."""

import unittest

from transformers import VitDetConfig
from transformers.testing_utils import is_flaky, require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import VitDetBackbone, VitDetModel


class VitDetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope

        self.num_patches_one_direction = self.image_size // self.patch_size
        self.seq_length = (self.image_size // self.patch_size) ** 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return VitDetConfig(
            image_size=self.image_size,
            pretrain_image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = VitDetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_size, self.num_patches_one_direction, self.num_patches_one_direction),
        )

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = VitDetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [self.batch_size, self.hidden_size, self.num_patches_one_direction, self.num_patches_one_direction],
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, [config.hidden_size])

        # verify backbone works with out_features=None
        config.out_features = None
        model = VitDetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [self.batch_size, self.hidden_size, self.num_patches_one_direction, self.num_patches_one_direction],
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VitDetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VitDet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VitDetModel, VitDetBackbone) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": VitDetModel} if is_torch_available() else {}

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = VitDetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VitDetConfig, has_text_modality=False, hidden_size=37)

    @is_flaky(max_attempts=3, description="`torch.nn.init.trunc_normal_` is flaky.")
    def test_initialization(self):
        super().test_initialization()

    # TODO: Fix me (once this model gets more usage)
    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        super().test_cpu_offload()

    # TODO: Fix me (once this model gets more usage)
    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_disk_offload_bin(self):
        super().test_disk_offload()

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_disk_offload_safetensors(self):
        super().test_disk_offload()

    # TODO: Fix me (once this model gets more usage)
    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        super().test_model_parallelism()

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VitDet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = self.model_tester.num_hidden_layers
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            # VitDet's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [
                    self.model_tester.num_patches_one_direction,
                    self.model_tester.num_patches_one_direction,
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # overwrite since VitDet only supports retraining gradients of hidden states
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

    @unittest.skip(reason="VitDet does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="VitDet does not have standalone checkpoints since it used as backbone in other models")
    def test_model_from_pretrained(self):
        pass


@require_torch
class VitDetBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (VitDetBackbone,) if is_torch_available() else ()
    config_class = VitDetConfig

    has_attentions = False

    def setUp(self):
        self.model_tester = VitDetModelTester(self)
