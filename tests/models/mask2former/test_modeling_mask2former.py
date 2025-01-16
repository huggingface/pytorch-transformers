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
"""Testing suite for the PyTorch Mask2Former model."""

import unittest

import numpy as np

from tests.test_modeling_common import floats_tensor
from transformers import Mask2FormerConfig, is_torch_available, is_vision_available
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4
from transformers.testing_utils import (
    require_timm,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_torch_multi_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerModel

    if is_vision_available():
        from transformers import Mask2FormerImageProcessor

if is_vision_available():
    from PIL import Image


class Mask2FormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        use_auxiliary_loss=False,
        num_queries=10,
        num_channels=3,
        min_size=32 * 8,
        max_size=32 * 8,
        num_labels=4,
        hidden_dim=64,
        num_attention_heads=4,
        num_hidden_layers=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.mask_feature_size = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size]).to(
            torch_device
        )

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        mask_labels = (
            torch.rand([self.batch_size, self.num_labels, self.min_size, self.max_size], device=torch_device) > 0.5
        ).float()
        class_labels = (torch.rand((self.batch_size, self.num_labels), device=torch_device) > 0.5).long()

        config = self.get_config()
        return config, pixel_values, pixel_mask, mask_labels, class_labels

    def get_config(self):
        config = Mask2FormerConfig(
            hidden_size=self.hidden_dim,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            encoder_feedforward_dim=16,
            dim_feedforward=32,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            decoder_layers=2,
            encoder_layers=2,
            feature_size=16,
        )
        config.num_queries = self.num_queries
        config.num_labels = self.num_labels

        config.backbone_config.embed_dim = 16
        config.backbone_config.depths = [1, 1, 1, 1]
        config.backbone_config.hidden_size = 16
        config.backbone_config.num_channels = self.num_channels
        config.backbone_config.num_heads = [1, 1, 2, 2]
        config.backbone = None

        config.hidden_dim = self.hidden_dim
        config.mask_feature_size = self.hidden_dim
        config.feature_size = self.hidden_dim
        return config

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def check_output_hidden_state(self, output, config):
        encoder_hidden_states = output.encoder_hidden_states
        pixel_decoder_hidden_states = output.pixel_decoder_hidden_states
        transformer_decoder_hidden_states = output.transformer_decoder_hidden_states

        self.parent.assertTrue(len(encoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(pixel_decoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(transformer_decoder_hidden_states), config.decoder_layers)

    def create_and_check_mask2former_model(self, config, pixel_values, pixel_mask, output_hidden_states=False):
        with torch.no_grad():
            model = Mask2FormerModel(config=config)
            model.to(torch_device)
            model.eval()

            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            output = model(pixel_values, output_hidden_states=True)

        self.parent.assertEqual(
            output.transformer_decoder_last_hidden_state.shape,
            (self.batch_size, self.num_queries, self.hidden_dim),
        )
        # let's ensure the other two hidden state exists
        self.parent.assertTrue(output.pixel_decoder_last_hidden_state is not None)
        self.parent.assertTrue(output.encoder_last_hidden_state is not None)

        if output_hidden_states:
            self.check_output_hidden_state(output, config)

    def create_and_check_mask2former_instance_segmentation_head_model(
        self, config, pixel_values, pixel_mask, mask_labels, class_labels
    ):
        model = Mask2FormerForUniversalSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        def comm_check_on_output(result):
            # let's still check that all the required stuff is there
            self.parent.assertTrue(result.transformer_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.pixel_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.encoder_last_hidden_state is not None)
            # okay, now we need to check the logits shape
            # due to the encoder compression, masks have a //4 spatial size
            self.parent.assertEqual(
                result.masks_queries_logits.shape,
                (self.batch_size, self.num_queries, self.min_size // 4, self.max_size // 4),
            )
            # + 1 for null class
            self.parent.assertEqual(
                result.class_queries_logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1)
            )

        with torch.no_grad():
            result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            result = model(pixel_values)

            comm_check_on_output(result)

            result = model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels
            )

        comm_check_on_output(result)

        self.parent.assertTrue(result.loss is not None)
        self.parent.assertEqual(result.loss.shape, torch.Size([]))


@require_torch
class Mask2FormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Mask2FormerModel, Mask2FormerForUniversalSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": Mask2FormerModel} if is_torch_available() else {}

    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = Mask2FormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Mask2FormerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mask2former_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_mask2former_model(config, **inputs, output_hidden_states=False)

    def test_mask2former_instance_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mask2former_instance_segmentation_head_model(*config_and_inputs)

    @unittest.skip(reason="Mask2Former does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Mask2Former does not have a get_input_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Mask2Former is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="Mask2Former does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(
        reason="Mask2Former has some layers using `add_module` which doesn't work well with `nn.DataParallel`"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["facebook/mask2former-swin-small-coco-instance"]:
            model = Mask2FormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_with_labels(self):
        size = (self.model_tester.min_size,) * 2
        inputs = {
            "pixel_values": torch.randn((2, 3, *size), device=torch_device),
            "mask_labels": torch.randn((2, 10, *size), device=torch_device),
            "class_labels": torch.zeros(2, 10, device=torch_device).long(),
        }
        config = self.model_tester.get_config()

        model = Mask2FormerForUniversalSegmentation(config).to(torch_device)
        outputs = model(**inputs)
        self.assertTrue(outputs.loss is not None)

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_mask2former_model(config, **inputs, output_hidden_states=True)

    def test_attention_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(**inputs, output_attentions=True)
            self.assertTrue(outputs.attentions is not None)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        model_class = self.all_model_classes[1]
        config, pixel_values, pixel_mask, mask_labels, class_labels = self.model_tester.prepare_config_and_inputs()

        model = model_class(config)
        model.to(torch_device)
        model.train()

        loss = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels).loss
        loss.backward()

    def test_retain_grad_hidden_states_attentions(self):
        model_class = self.all_model_classes[1]
        config, pixel_values, pixel_mask, mask_labels, class_labels = self.model_tester.prepare_config_and_inputs()
        config.output_hidden_states = True
        config.output_attentions = True

        model = model_class(config).to(torch_device)
        model.train()

        outputs = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states[0]
        pixel_decoder_hidden_states.retain_grad()

        transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states[0]
        transformer_decoder_hidden_states.retain_grad()

        attentions = outputs.attentions[0]
        attentions.retain_grad()

        outputs.loss.backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(pixel_decoder_hidden_states.grad)
        self.assertIsNotNone(transformer_decoder_hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

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
            if model.__class__.__name__ == "Mask2FormerModel":
                self.assertEqual(model.pixel_level_module.encoder.out_indices, [1, 2, 3])
            elif model.__class__.__name__ == "Mask2FormerForUniversalSegmentation":
                self.assertEqual(model.model.pixel_level_module.encoder.out_indices, [1, 2, 3])

        # Load a HF backbone
        config.backbone = "microsoft/resnet-18"
        config.use_timm_backbone = False

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            if model.__class__.__name__ == "Mask2FormerModel":
                self.assertEqual(model.pixel_level_module.encoder.out_indices, [1, 2, 3])
            elif model.__class__.__name__ == "Mask2FormerForUniversalSegmentation":
                self.assertEqual(model.model.pixel_level_module.encoder.out_indices, [1, 2, 3])


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@slow
class Mask2FormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def model_checkpoints(self):
        return "facebook/mask2former-swin-small-coco-instance"

    @cached_property
    def default_image_processor(self):
        return Mask2FormerImageProcessor.from_pretrained(self.model_checkpoints) if is_vision_available() else None

    def test_inference_no_head(self):
        model = Mask2FormerModel.from_pretrained(self.model_checkpoints).to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size is divisible by 32
        self.assertTrue((inputs_shape[-1] % 32) == 0 and (inputs_shape[-2] % 32) == 0)
        # check size
        self.assertEqual(inputs_shape, (1, 3, 384, 384))

        with torch.no_grad():
            outputs = model(**inputs)

        expected_slice_hidden_state = torch.tensor(
            [[-0.2790, -1.0717, -1.1668], [-0.5128, -0.3128, -0.4987], [-0.5832, 0.1971, -0.0197]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.encoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[0.8973, 1.1847, 1.1776], [1.1934, 1.5040, 1.5128], [1.1153, 1.4486, 1.4951]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.pixel_decoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[2.1152, 1.7000, -0.8603], [1.5808, 1.8004, -0.9353], [1.6043, 1.7495, -0.5999]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.transformer_decoder_last_hidden_state[0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

    def test_inference_universal_segmentation_head(self):
        model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_checkpoints).to(torch_device).eval()
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size is divisible by 32
        self.assertTrue((inputs_shape[-1] % 32) == 0 and (inputs_shape[-2] % 32) == 0)
        # check size
        self.assertEqual(inputs_shape, (1, 3, 384, 384))

        with torch.no_grad():
            outputs = model(**inputs)
        # masks_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        self.assertEqual(
            masks_queries_logits.shape, (1, model.config.num_queries, inputs_shape[-2] // 4, inputs_shape[-1] // 4)
        )
        expected_slice = [
            [-8.7839, -9.0056, -8.8121],
            [-7.4104, -7.0313, -6.5401],
            [-6.6105, -6.3427, -6.4675],
        ]
        expected_slice = torch.tensor(expected_slice).to(torch_device)
        torch.testing.assert_close(masks_queries_logits[0, 0, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)
        # class_queries_logits
        class_queries_logits = outputs.class_queries_logits
        self.assertEqual(class_queries_logits.shape, (1, model.config.num_queries, model.config.num_labels + 1))
        expected_slice = torch.tensor(
            [
                [1.8324, -8.0835, -4.1922],
                [0.8450, -9.0050, -3.6053],
                [0.3045, -7.7293, -3.0275],
            ]
        ).to(torch_device)
        torch.testing.assert_close(
            outputs.class_queries_logits[0, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE
        )

    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(self.model_checkpoints)
            .to(torch_device, dtype=torch.float16)
            .eval()
        )
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        with torch.no_grad():
            _ = model(**inputs)

    def test_with_segmentation_maps_and_loss(self):
        model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_checkpoints).to(torch_device).eval()
        image_processor = self.default_image_processor

        inputs = image_processor(
            [np.zeros((3, 800, 1333)), np.zeros((3, 800, 1333))],
            segmentation_maps=[np.zeros((384, 384)).astype(np.float32), np.zeros((384, 384)).astype(np.float32)],
            return_tensors="pt",
        )

        inputs["pixel_values"] = inputs["pixel_values"].to(torch_device)
        inputs["mask_labels"] = [el.to(torch_device) for el in inputs["mask_labels"]]
        inputs["class_labels"] = [el.to(torch_device) for el in inputs["class_labels"]]

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(outputs.loss is not None)

    def test_export(self):
        if not is_torch_greater_or_equal_than_2_4:
            self.skipTest(reason="This test requires torch >= 2.4 to run.")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_checkpoints).to(torch_device).eval()
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        exported_program = torch.export.export(
            model,
            args=(inputs["pixel_values"], inputs["pixel_mask"]),
            strict=True,
        )
        with torch.no_grad():
            eager_outputs = model(**inputs)
            exported_outputs = exported_program.module().forward(inputs["pixel_values"], inputs["pixel_mask"])
        self.assertEqual(eager_outputs.masks_queries_logits.shape, exported_outputs.masks_queries_logits.shape)
        torch.testing.assert_close(
            eager_outputs.masks_queries_logits, exported_outputs.masks_queries_logits, rtol=TOLERANCE, atol=TOLERANCE
        )
        self.assertEqual(eager_outputs.class_queries_logits.shape, exported_outputs.class_queries_logits.shape)
        torch.testing.assert_close(
            eager_outputs.class_queries_logits, exported_outputs.class_queries_logits, rtol=TOLERANCE, atol=TOLERANCE
        )
