# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch VitPose backbone model."""

import inspect
import unittest

from transformers import VitPoseBackboneConfig
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available, is_vision_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    from transformers import VitPoseBackbone


if is_vision_available():
    pass


class VitPoseBackboneModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=[16 * 8, 12 * 8],
        patch_size=[8, 8],
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=2,
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
        self.num_labels = num_labels
        self.scope = scope

        # in VitPoseBackbone, the seq length equals the number of patches
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.seq_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return VitPoseBackboneConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VitPoseBackboneModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VitPoseBackbone does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VitPoseBackbone,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = VitPoseBackboneModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=VitPoseBackboneConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VitPoseBackbone does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support input and output embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not output a loss")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support training yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="VitPoseBackbone does not support training yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)


@require_torch
class VitPoseBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (VitPoseBackbone,) if is_torch_available() else ()
    config_class = VitPoseBackboneConfig

    has_attentions = False

    def setUp(self):
        self.model_tester = VitPoseBackboneModelTester(self)
