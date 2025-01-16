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
"""Testing suite for the PyTorch Pixtral model."""

import unittest

from transformers import (
    PixtralVisionConfig,
    PixtralVisionModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


class PixtralVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return PixtralVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = PixtralVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, pixel_values):
        model = PixtralVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.image_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class PixtralVisionModelModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `PixtralVisionModel`.
    """

    all_model_classes = (PixtralVisionModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = PixtralVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PixtralVisionConfig, has_text_modality=False)

    @unittest.skip("model does not support input embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("model does not support input embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Pixtral models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Pixtral models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_save_load(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_model_main_input_name(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_gradient_checkpointing_backward_compatibility(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Not supported yet")
    def test_determinism(self):
        pass
