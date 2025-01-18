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
"""Testing suite for the PyTorch Prompt Depth Anything model."""

import unittest

import requests

from transformers import Dinov2Config, PromptDepthAnythingConfig
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import PromptDepthAnythingForDepthEstimation


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class PromptDepthAnythingModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
        use_labels=True,
        num_labels=3,
        is_training=True,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        out_features=["stage1", "stage2"],
        apply_layernorm=False,
        reshape_hidden_states=False,
        neck_hidden_sizes=[2, 2],
        fusion_hidden_size=6,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.out_features = out_features
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.is_training = is_training
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.seq_length = (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        prompt_depth = floats_tensor([self.batch_size, 1, self.image_size // 4, self.image_size // 4])

        config = self.get_config()

        return config, pixel_values, labels, prompt_depth

    def get_config(self):
        return PromptDepthAnythingConfig(
            backbone_config=self.get_backbone_config(),
            reassemble_hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            neck_hidden_sizes=self.neck_hidden_sizes,
            fusion_hidden_size=self.fusion_hidden_size,
        )

    def get_backbone_config(self):
        return Dinov2Config(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            is_training=self.is_training,
            out_features=self.out_features,
            reshape_hidden_states=self.reshape_hidden_states,
        )

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels, prompt_depth):
        config.num_labels = self.num_labels
        model = PromptDepthAnythingForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, prompt_depth=prompt_depth)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, prompt_depth = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "prompt_depth": prompt_depth}
        return config, inputs_dict


@require_torch
class PromptDepthAnythingModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Prompt Depth Anything does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (PromptDepthAnythingForDepthEstimation,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"depth-estimation": PromptDepthAnythingForDepthEstimation} if is_torch_available() else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = PromptDepthAnythingModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PromptDepthAnythingConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["patch_size"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(
        reason="Prompt Depth Anything with AutoBackbone does not have a base model and hence no input_embeddings"
    )
    def test_inputs_embeds(self):
        pass

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    @unittest.skip(reason="Prompt Depth Anything does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Prompt Depth Anything does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="Prompt Depth Anything with AutoBackbone does not have a base model and hence no input_embeddings"
    )
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Prompt Depth Anything with AutoBackbone does not have a base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Prompt Depth Anything with AutoBackbone does not have a base model")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "depth-anything/prompt-depth-anything-vits-hf"
        model = PromptDepthAnythingForDepthEstimation.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_backbone_selection(self):
        def _validate_backbone_init():
            for model_class in self.all_model_classes:
                model = model_class(config)
                model.to(torch_device)
                model.eval()

                self.assertEqual(len(model.backbone.out_indices), 2)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.backbone = "facebook/dinov2-small"
        config.use_pretrained_backbone = True
        config.use_timm_backbone = False
        config.backbone_config = None
        config.backbone_kwargs = {"out_indices": [-2, -1]}
        _validate_backbone_init()


def prepare_img():
    url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def prepare_prompt_depth():
    prompt_depth_url = (
        "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
    )
    prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)
    return prompt_depth


@require_torch
@require_vision
@slow
class PromptDepthAnythingModelIntegrationTest(unittest.TestCase):
    def test_inference_wo_prompt_depth(self):
        image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
        model = PromptDepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/prompt-depth-anything-vits-hf"
        ).to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        expected_shape = torch.Size([1, 756, 1008])
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.5029, 0.5120, 0.5176], [0.4998, 0.5147, 0.5197], [0.4973, 0.5201, 0.5241]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-3))

    def test_inference(self):
        image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
        model = PromptDepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/prompt-depth-anything-vits-hf"
        ).to(torch_device)

        image = prepare_img()
        prompt_depth = prepare_prompt_depth()
        inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        expected_shape = torch.Size([1, 756, 1008])
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[3.0100, 3.0016, 3.0219], [3.0046, 3.0137, 3.0275], [3.0083, 3.0191, 3.0292]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-3))

    def test_export(self):
        for strict in [True, False]:
            with self.subTest(strict=strict):
                if not is_torch_greater_or_equal_than_2_4:
                    self.skipTest(reason="This test requires torch >= 2.4 to run.")
                model = (
                    PromptDepthAnythingForDepthEstimation.from_pretrained(
                        "depth-anything/prompt-depth-anything-vits-hf"
                    )
                    .to(torch_device)
                    .eval()
                )
                image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
                image = prepare_img()
                prompt_depth = prepare_prompt_depth()
                inputs = image_processor(images=image, prompt_depth=prompt_depth, return_tensors="pt").to(torch_device)

                exported_program = torch.export.export(
                    model,
                    args=(inputs["pixel_values"], inputs["prompt_depth"]),
                    strict=strict,
                )
                with torch.no_grad():
                    eager_outputs = model(**inputs)
                    exported_outputs = exported_program.module().forward(inputs["pixel_values"])
                self.assertEqual(eager_outputs.predicted_depth.shape, exported_outputs.predicted_depth.shape)
                self.assertTrue(
                    torch.allclose(eager_outputs.predicted_depth, exported_outputs.predicted_depth, atol=1e-4)
                )
