# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch DepthPro model."""

import unittest

from transformers import DepthProConfig
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DepthProForDepthEstimation, DepthProModel
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


if is_vision_available():
    from PIL import Image

    from transformers import DepthProImageProcessor


class DepthProModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        image_size=64,
        patch_size=8,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        fusion_hidden_size=16,
        intermediate_hook_ids=[0],
        intermediate_feature_dims=[8],
        scaled_images_ratios=[0.5, 1.0],
        scaled_images_overlap_ratios=[0.0, 0.2],
        scaled_images_feature_dims=[12, 12],
        num_hidden_layers=1,
        num_attention_heads=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        use_fov_model=False,
        backbone_config={
            "model_type": "dinov2",
            "patch_size": 4,
        },
        num_labels=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.intermediate_hook_ids = intermediate_hook_ids
        self.intermediate_feature_dims = intermediate_feature_dims
        self.scaled_images_ratios = scaled_images_ratios
        self.scaled_images_overlap_ratios = scaled_images_overlap_ratios
        self.scaled_images_feature_dims = scaled_images_feature_dims
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.use_fov_model = use_fov_model
        self.backbone_config = backbone_config
        self.num_labels = num_labels

        # may be different for a backbone other than dinov2
        self.out_size = patch_size // backbone_config["patch_size"]
        self.seq_length = self.out_size**2 + 1  # we add 1 for the [CLS] token

        n_fusion_blocks = len(intermediate_hook_ids) + len(scaled_images_ratios)
        self.expected_depth_size = 2 ** (n_fusion_blocks + 1) * self.out_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DepthProConfig(
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            fusion_hidden_size=self.fusion_hidden_size,
            intermediate_hook_ids=self.intermediate_hook_ids,
            intermediate_feature_dims=self.intermediate_feature_dims,
            scaled_images_ratios=self.scaled_images_ratios,
            scaled_images_overlap_ratios=self.scaled_images_overlap_ratios,
            scaled_images_feature_dims=self.scaled_images_feature_dims,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            use_fov_model=self.use_fov_model,
            backbone_config=self.backbone_config,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DepthProModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # TODO: return hidden_states from patch_encodings instead of image_encodings
        # num_patches = result.last_hidden_state.shape[1]  # num_patches are created dynamically
        # self.parent.assertEqual(
        #     result.last_hidden_state.shape, (self.batch_size, num_patches, self.seq_length, self.hidden_size)
        # )
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DepthProForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.predicted_depth.shape, (self.batch_size, self.expected_depth_size, self.expected_depth_size)
        )

    def create_and_check_for_fov(self, config, pixel_values, labels):
        model = DepthProForDepthEstimation(config, use_fov_model=True)
        model.to(torch_device)
        model.eval()

        # check if the fov_model (DinoV2-based encoder) is created
        self.parent.assertIsNotNone(model.fov_model)

        batched_pixel_values = pixel_values
        row_pixel_values = pixel_values[:1]

        with torch.no_grad():
            model_batched_output_fov = model(batched_pixel_values).fov
            model_row_output_fov = model(row_pixel_values).fov

        # check if fov is returned
        self.parent.assertIsNotNone(model_batched_output_fov)
        self.parent.assertIsNotNone(model_row_output_fov)

        # check output shape consistency for fov
        self.parent.assertEqual(model_batched_output_fov.shape, (self.batch_size,))

        # check equivalence between batched and single row outputs for fov
        diff = torch.max(torch.abs(model_row_output_fov - model_batched_output_fov[:1]))
        model_name = model.__class__.__name__
        self.parent.assertTrue(
            diff <= 1e-03,
            msg=(f"Batched and Single row outputs are not equal in {model_name} for fov. " f"Difference={diff}."),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DepthProModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as DepthPro does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DepthProModel, DepthProForDepthEstimation) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "depth-estimation": DepthProForDepthEstimation,
            "image-feature-extraction": DepthProModel,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DepthProModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DepthProConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="DepthPro does not use inputs_embeds")
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

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    def test_for_fov(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_fov(*config_and_inputs)

    def test_training(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DepthProForDepthEstimation":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in MODEL_MAPPING_NAMES.values():
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DepthProForDepthEstimation":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            if model_class.__name__ in MODEL_MAPPING_NAMES.values() or not model_class.supports_gradient_checkpointing:
                continue
            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            # Skip the check for the backbone
            backbone_params = []
            for name, module in model.named_modules():
                if module.__class__.__name__ == "DepthProViTHybridEmbeddings":
                    backbone_params = [f"{name}.{key}" for key in module.state_dict().keys()]
                    break

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in backbone_params:
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @slow
    def test_model_from_pretrained(self):
        model_path = "geetu040/DepthPro"
        model = DepthProModel.from_pretrained(model_path)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DepthProModelIntegrationTest(unittest.TestCase):
    def test_inference_depth_estimation(self):
        model_path = "geetu040/DepthPro"
        image_processor = DepthProImageProcessor.from_pretrained(model_path)
        model = DepthProForDepthEstimation.from_pretrained(model_path).to(torch_device)
        config = model.config

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        n_fusion_blocks = len(config.intermediate_hook_ids) + len(config.scaled_images_ratios)
        out_size = config.backbone_config.image_size // config.backbone_config.patch_size
        expected_depth_size = 2 ** (n_fusion_blocks + 1) * out_size
        expected_shape = torch.Size((1, expected_depth_size, expected_depth_size))
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[1.0582, 1.1225, 1.1335], [1.1154, 1.1398, 1.1486], [1.1434, 1.1500, 1.1643]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-4))

    def test_post_processing_depth_estimation(self):
        model_path = "geetu040/DepthPro"
        image_processor = DepthProImageProcessor.from_pretrained(model_path)
        model = DepthProForDepthEstimation.from_pretrained(model_path)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        outputs = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[[image.height, image.width]],
        )
        predicted_depth = outputs[0]["predicted_depth"]
        expected_shape = torch.Size((image.height, image.width))
        self.assertTrue(predicted_depth.shape == expected_shape)
