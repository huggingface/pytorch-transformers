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
"""Testing suite for the PyTorch UperNet framework."""

import unittest

from huggingface_hub import hf_hub_download

from transformers import ConvNextConfig, UperNetConfig
from transformers.testing_utils import (
    require_timm,
    require_torch,
    require_torch_multi_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import UperNetForSemanticSegmentation


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class UperNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        num_channels=3,
        num_stages=4,
        hidden_sizes=[10, 20, 30, 40],
        depths=[1, 1, 1, 1],
        is_training=True,
        use_labels=True,
        intermediate_size=37,
        hidden_act="gelu",
        type_sequence_label_size=10,
        initializer_range=0.02,
        out_features=["stage2", "stage3", "stage4"],
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.out_features = out_features
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_backbone_config(self):
        return ConvNextConfig(
            num_channels=self.num_channels,
            num_stages=self.num_stages,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            is_training=self.is_training,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
        )

    def get_config(self):
        return UperNetConfig(
            backbone_config=self.get_backbone_config(),
            backbone=None,
            hidden_size=64,
            pool_scales=[1, 2, 3, 6],
            use_auxiliary_head=True,
            auxiliary_loss_weight=0.4,
            auxiliary_in_channels=40,
            auxiliary_channels=32,
            auxiliary_num_convs=1,
            auxiliary_concat_input=False,
            loss_ignore_index=255,
            num_labels=self.num_labels,
        )

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels):
        model = UperNetForSemanticSegmentation(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size, self.image_size)
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
class UperNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as UperNet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (UperNetForSemanticSegmentation,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-segmentation": UperNetForSemanticSegmentation} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    has_attentions = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = UperNetModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=UperNetConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["hidden_size"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    @unittest.skip(reason="UperNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="UperNet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="UperNet does not have a base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="UperNet does not have a base model")
    def test_save_load_fast_init_to_base(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(reason="UperNet has some layers using `add_module` which doesn't work well with `nn.DataParallel`")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_stages = self.model_tester.num_stages
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            # ConvNext's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 4, self.model_tester.image_size // 4],
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
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        configs_no_init.backbone_config = _config_zero_init(configs_no_init.backbone_config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @require_timm
    def test_backbone_selection(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        config.backbone_config = None
        config.backbone_kwargs = {"out_indices": [1, 2, 3]}
        config.use_pretrained_backbone = True

        # Load a timm backbone
        # We can't load transformer checkpoint with timm backbone, as we can't specify features_only and out_indices
        config.backbone = "resnet18"
        config.use_timm_backbone = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            if model.__class__.__name__ == "UperNetForUniversalSegmentation":
                self.assertEqual(model.backbone.out_indices, [1, 2, 3])

        # Load a HF backbone
        config.backbone = "microsoft/resnet-18"
        config.use_timm_backbone = False

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            if model.__class__.__name__ == "UperNetForUniversalSegmentation":
                self.assertEqual(model.backbone.out_indices, [1, 2, 3])

    @unittest.skip(reason="UperNet does not have tied weights")
    def test_tied_model_weights_key_ignore(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "openmmlab/upernet-convnext-tiny"
        model = UperNetForSemanticSegmentation.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of ADE20k
def prepare_img():
    filepath = hf_hub_download(
        repo_id="hf-internal-testing/fixtures_ade20k", repo_type="dataset", filename="ADE_val_00000001.jpg"
    )
    image = Image.open(filepath).convert("RGB")
    return image


@require_torch
@require_vision
@slow
class UperNetModelIntegrationTest(unittest.TestCase):
    def test_inference_swin_backbone(self):
        processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-tiny")
        model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-tiny").to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, model.config.num_labels, 512, 512))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-7.5958, -7.5958, -7.4302], [-7.5958, -7.5958, -7.4302], [-7.4797, -7.4797, -7.3068]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_convnext_backbone(self):
        processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
        model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny").to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, model.config.num_labels, 512, 512))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-8.8110, -8.8110, -8.6521], [-8.8110, -8.8110, -8.6521], [-8.7746, -8.7746, -8.6130]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))
