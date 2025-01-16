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
"""Testing suite for the PyTorch VitMatte model."""

import unittest

from huggingface_hub import hf_hub_download

from transformers import VitMatteConfig
from transformers.testing_utils import (
    require_timm,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import VitDetConfig, VitMatteForImageMatting


if is_vision_available():
    from PIL import Image

    from transformers import VitMatteImageProcessor


class VitMatteModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=16,
        num_channels=4,
        is_training=True,
        use_labels=False,
        hidden_size=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_act="gelu",
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        out_features=["stage1"],
        fusion_hidden_sizes=[128, 64, 32, 16],
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
        self.hidden_act = hidden_act
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.out_features = out_features
        self.fusion_hidden_sizes = fusion_hidden_sizes

        self.seq_length = (self.image_size // self.patch_size) ** 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            raise NotImplementedError("Training is not yet supported")

        config = self.get_config()

        return config, pixel_values, labels

    def get_backbone_config(self):
        return VitDetConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            is_training=self.is_training,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
        )

    def get_config(self):
        return VitMatteConfig(
            backbone_config=self.get_backbone_config(),
            backbone=None,
            hidden_size=self.hidden_size,
            fusion_hidden_sizes=self.fusion_hidden_sizes,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = VitMatteForImageMatting(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.alphas.shape, (self.batch_size, 1, self.image_size, self.image_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VitMatteModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VitMatte does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VitMatteForImageMatting,) if is_torch_available() else ()
    pipeline_model_mapping = {}

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = VitMatteModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VitMatteConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["hidden_size"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VitMatte does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
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

    @unittest.skip(reason="ViTMatte does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "hustvl/vitmatte-small-composition-1k"
        model = VitMatteForImageMatting.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="ViTMatte does not support retaining gradient on attention logits")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [2, 2],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            print("Hello we're here")

            check_hidden_states_output(inputs_dict, config, model_class)

    @require_timm
    def test_backbone_selection(self):
        def _validate_backbone_init():
            for model_class in self.all_model_classes:
                model = model_class(config)
                model.to(torch_device)
                model.eval()

                if model.__class__.__name__ == "VitMatteForImageMatting":
                    # Confirm out_indices propogated to backbone
                    self.assertEqual(len(model.backbone.out_indices), 2)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.use_pretrained_backbone = True
        config.backbone_config = None
        config.backbone_kwargs = {"out_indices": [-2, -1]}
        # Force load_backbone path
        config.is_hybrid = False

        # Load a timm backbone
        config.backbone = "resnet18"
        config.use_timm_backbone = True
        _validate_backbone_init()

        # Load a HF backbone
        config.backbone = "facebook/dinov2-small"
        config.use_timm_backbone = False
        _validate_backbone_init()


@require_torch
class VitMatteModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k").to(torch_device)

        filepath = hf_hub_download(
            repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        )
        image = Image.open(filepath).convert("RGB")
        filepath = hf_hub_download(
            repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        )
        trimap = Image.open(filepath).convert("L")

        # prepare image + trimap for the model
        inputs = processor(images=image, trimaps=trimap, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            alphas = model(**inputs).alphas

        expected_shape = torch.Size((1, 1, 640, 960))
        self.assertEqual(alphas.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.9977, 0.9987, 0.9990], [0.9980, 0.9998, 0.9998], [0.9983, 0.9998, 0.9998]], device=torch_device
        )
        torch.testing.assert_close(alphas[0, 0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
