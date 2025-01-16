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
"""Testing suite for the PyTorch OneFormer model."""

import copy
import inspect
import unittest

import numpy as np

from tests.test_modeling_common import floats_tensor
from transformers import OneFormerConfig, is_torch_available, is_vision_available
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

    from transformers import OneFormerForUniversalSegmentation, OneFormerModel

    if is_vision_available():
        from transformers import OneFormerProcessor

if is_vision_available():
    from PIL import Image


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


class OneFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        vocab_size=99,
        use_auxiliary_loss=False,
        num_queries=10,
        num_channels=3,
        min_size=32 * 8,
        max_size=32 * 8,
        num_labels=4,
        hidden_dim=64,
        sequence_length=77,
        n_ctx=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.use_auxiliary_loss = use_auxiliary_loss
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.n_ctx = n_ctx

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size]).to(
            torch_device
        )

        task_inputs = (
            torch.randint(high=self.vocab_size, size=(self.batch_size, self.sequence_length)).to(torch_device).long()
        )

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        text_inputs = (
            torch.randint(
                high=self.vocab_size, size=(self.batch_size, self.num_queries - self.n_ctx, self.sequence_length)
            )
            .to(torch_device)
            .long()
        )

        mask_labels = (
            torch.rand([self.batch_size, self.num_labels, self.min_size, self.max_size], device=torch_device) > 0.5
        ).float()
        class_labels = (torch.rand((self.batch_size, self.num_labels), device=torch_device) > 0.5).long()

        config = self.get_config()
        return config, pixel_values, task_inputs, text_inputs, pixel_mask, mask_labels, class_labels

    def get_config(self):
        config = OneFormerConfig(
            text_encoder_vocab_size=self.vocab_size,
            hidden_size=self.hidden_dim,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            encoder_feedforward_dim=32,
            dim_feedforward=64,
            encoder_layers=2,
            decoder_layers=2,
        )

        config.backbone_config.embed_dim = 16
        config.backbone_config.depths = [1, 1, 1, 1]
        config.backbone_config.hidden_size = 16
        config.backbone_config.num_channels = self.num_channels
        config.backbone_config.num_heads = [1, 1, 2, 2]
        config.backbone = None

        config.hidden_dim = self.hidden_dim
        config.mask_dim = self.hidden_dim
        config.conv_dim = self.hidden_dim

        config.text_encoder_width = self.hidden_dim
        config.task_seq_len = self.sequence_length
        config.max_seq_len = self.sequence_length
        config.text_encoder_context_length = self.sequence_length
        config.text_encoder_n_ctx = self.n_ctx

        return config

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, task_inputs, pixel_mask, _, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "task_inputs": task_inputs}
        return config, inputs_dict

    def check_output_hidden_state(self, output, config):
        encoder_hidden_states = output.encoder_hidden_states
        pixel_decoder_hidden_states = output.pixel_decoder_hidden_states
        transformer_decoder_hidden_states = output.transformer_decoder_hidden_states

        self.parent.assertTrue(len(encoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(pixel_decoder_hidden_states), config.encoder_layers)
        self.parent.assertTrue(len(transformer_decoder_hidden_states), config.decoder_layers - 1)

    def create_and_check_oneformer_model(
        self, config, pixel_values, task_inputs, pixel_mask, output_hidden_states=False
    ):
        with torch.no_grad():
            model = OneFormerModel(config=config)
            model.to(torch_device)
            model.eval()

            output = model(pixel_values=pixel_values, task_inputs=task_inputs, pixel_mask=pixel_mask)
            output = model(pixel_values, task_inputs=task_inputs, output_hidden_states=True)
        # the correct shape of output.transformer_decoder_hidden_states ensure the correcteness of the
        # encoder and pixel decoder
        self.parent.assertEqual(
            output.transformer_decoder_object_queries.shape,
            (self.batch_size, self.num_queries, self.hidden_dim),
        )
        # let's ensure the other two hidden state exists
        self.parent.assertTrue(output.pixel_decoder_hidden_states is not None)
        self.parent.assertTrue(output.encoder_hidden_states is not None)

        if output_hidden_states:
            self.check_output_hidden_state(output, config)

    def create_and_check_oneformer_universal_segmentation_head_model(
        self, config, pixel_values, task_inputs, text_inputs, pixel_mask, mask_labels, class_labels
    ):
        model = OneFormerForUniversalSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        def comm_check_on_output(result):
            # let's still check that all the required stuff is there
            self.parent.assertTrue(result.transformer_decoder_hidden_states is not None)
            self.parent.assertTrue(result.pixel_decoder_hidden_states is not None)
            self.parent.assertTrue(result.encoder_hidden_states is not None)
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
            result = model(pixel_values=pixel_values, task_inputs=task_inputs, pixel_mask=pixel_mask)
            result = model(pixel_values, task_inputs)

            comm_check_on_output(result)

        config.is_training = True
        model = OneFormerForUniversalSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(
                pixel_values=pixel_values,
                task_inputs=task_inputs,
                pixel_mask=pixel_mask,
                mask_labels=mask_labels,
                class_labels=class_labels,
                text_inputs=text_inputs,
            )

        comm_check_on_output(result)

        self.parent.assertTrue(result.loss is not None)
        self.parent.assertEqual(result.loss.shape, torch.Size([1]))


@require_torch
class OneFormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (OneFormerModel, OneFormerForUniversalSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": OneFormerModel} if is_torch_available() else {}

    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    # TODO: Fix the failed tests when this model gets more usage
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if pipeline_test_case_name == "FeatureExtractionPipelineTests":
            return True

        return False

    def setUp(self):
        self.model_tester = OneFormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OneFormerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_oneformer_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_oneformer_model(config, **inputs, output_hidden_states=False)

    def test_oneformer_universal_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_oneformer_universal_segmentation_head_model(*config_and_inputs)

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1:3]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    @unittest.skip(reason="OneFormer uses two main inputs")
    def test_torchscript_simple(self):
        pass

    @unittest.skip(reason="OneFormer uses two main inputs")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="OneFormer uses two main inputs")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip(reason="OneFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="OneFormer does not have a get_input_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="OneFormer is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="OneFormer does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(
        reason="OneFormer has some layers using `add_module` which doesn't work well with `nn.DataParallel`"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values", "task_inputs"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["shi-labs/oneformer_ade20k_swin_tiny"]:
            model = OneFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_with_labels(self):
        size = (self.model_tester.min_size,) * 2
        inputs = {
            "pixel_values": torch.randn((2, 3, *size), device=torch_device),
            "task_inputs": torch.randint(high=self.model_tester.vocab_size, size=(2, 77), device=torch_device).long(),
            "text_inputs": torch.randint(
                high=self.model_tester.vocab_size, size=(2, 6, 77), device=torch_device
            ).long(),
            "mask_labels": torch.randn((2, 150, *size), device=torch_device),
            "class_labels": torch.zeros(2, 150, device=torch_device).long(),
        }

        config = self.model_tester.get_config()
        config.is_training = True

        model = OneFormerForUniversalSegmentation(config).to(torch_device)
        outputs = model(**inputs)
        self.assertTrue(outputs.loss is not None)

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_oneformer_model(config, **inputs, output_hidden_states=True)

    def test_attention_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(**inputs, output_attentions=True)
            self.assertTrue(outputs.attentions is not None)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.contrastive_temperature = 1

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")
        # only OneFormerForUniversalSegmentation has the loss
        model_class = self.all_model_classes[1]
        (
            config,
            pixel_values,
            task_inputs,
            text_inputs,
            pixel_mask,
            mask_labels,
            class_labels,
        ) = self.model_tester.prepare_config_and_inputs()
        config.is_training = True

        model = model_class(config)
        model.to(torch_device)
        model.train()

        loss = model(
            pixel_values, task_inputs, text_inputs=text_inputs, mask_labels=mask_labels, class_labels=class_labels
        ).loss
        loss.backward()

    def test_retain_grad_hidden_states_attentions(self):
        # only OneFormerForUniversalSegmentation has the loss
        model_class = self.all_model_classes[1]
        (
            config,
            pixel_values,
            task_inputs,
            text_inputs,
            pixel_mask,
            mask_labels,
            class_labels,
        ) = self.model_tester.prepare_config_and_inputs()
        config.output_hidden_states = True
        config.output_attentions = True
        config.is_training = True

        model = model_class(config)
        model.to(torch_device)
        model.train()

        outputs = model(
            pixel_values, task_inputs, text_inputs=text_inputs, mask_labels=mask_labels, class_labels=class_labels
        )

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states[0]
        pixel_decoder_hidden_states.retain_grad()

        transformer_decoder_class_predictions = outputs.transformer_decoder_class_predictions
        transformer_decoder_class_predictions.retain_grad()

        transformer_decoder_mask_predictions = outputs.transformer_decoder_mask_predictions
        transformer_decoder_mask_predictions.retain_grad()

        attentions = outputs.attentions[0][0]
        attentions.retain_grad()

        outputs.loss.backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(pixel_decoder_hidden_states.grad)
        self.assertIsNotNone(transformer_decoder_class_predictions.grad)
        self.assertIsNotNone(transformer_decoder_mask_predictions.grad)
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
            if model.__class__.__name__ == "OneFormerModel":
                self.assertEqual(model.pixel_level_module.encoder.out_indices, [1, 2, 3])
            elif model.__class__.__name__ == "OneFormerForUniversalSegmentation":
                self.assertEqual(model.model.pixel_level_module.encoder.out_indices, [1, 2, 3])

        # Load a HF backbone
        config.backbone = "microsoft/resnet-18"
        config.use_timm_backbone = False

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            if model.__class__.__name__ == "OneFormerModel":
                self.assertEqual(model.pixel_level_module.encoder.out_indices, [1, 2, 3])
            elif model.__class__.__name__ == "OneFormerForUniversalSegmentation":
                self.assertEqual(model.model.pixel_level_module.encoder.out_indices, [1, 2, 3])


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@slow
class OneFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def model_checkpoints(self):
        return "shi-labs/oneformer_ade20k_swin_tiny"

    @cached_property
    def default_processor(self):
        return OneFormerProcessor.from_pretrained(self.model_checkpoints) if is_vision_available() else None

    def test_inference_no_head(self):
        model = OneFormerModel.from_pretrained(self.model_checkpoints).to(torch_device)
        processor = self.default_processor
        image = prepare_img()
        inputs = processor(image, ["semantic"], return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size
        self.assertEqual(inputs_shape, (1, 3, 512, 682))

        task_inputs_shape = inputs["task_inputs"].shape
        # check size
        self.assertEqual(task_inputs_shape, (1, 77))

        with torch.no_grad():
            outputs = model(**inputs)

        expected_slice_hidden_state = torch.tensor(
            [[0.2723, 0.8280, 0.6026], [1.2699, 1.1257, 1.1444], [1.1344, 0.6153, 0.4177]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.encoder_hidden_states[-1][0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[1.0581, 1.2276, 1.2003], [1.1903, 1.2925, 1.2862], [1.158, 1.2559, 1.3216]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.pixel_decoder_hidden_states[0][0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[3.0668, -1.1833, -5.1103], [3.344, -3.362, -5.1101], [2.6017, -4.3613, -4.1444]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.transformer_decoder_class_predictions[0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

    def test_inference_universal_segmentation_head(self):
        model = OneFormerForUniversalSegmentation.from_pretrained(self.model_checkpoints).to(torch_device).eval()
        processor = self.default_processor
        image = prepare_img()
        inputs = processor(image, ["semantic"], return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size
        self.assertEqual(inputs_shape, (1, 3, 512, 682))

        with torch.no_grad():
            outputs = model(**inputs)

        # masks_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        self.assertEqual(
            masks_queries_logits.shape,
            (1, model.config.num_queries, inputs_shape[-2] // 4, (inputs_shape[-1] + 2) // 4),
        )
        expected_slice = [[[3.1848, 4.2141, 4.1993], [2.9000, 3.5721, 3.6603], [2.5358, 3.0883, 3.6168]]]
        expected_slice = torch.tensor(expected_slice).to(torch_device)
        torch.testing.assert_close(masks_queries_logits[0, 0, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)
        # class_queries_logits
        class_queries_logits = outputs.class_queries_logits
        self.assertEqual(
            class_queries_logits.shape,
            (1, model.config.num_queries, model.config.num_labels + 1),
        )
        expected_slice = torch.tensor(
            [[3.0668, -1.1833, -5.1103], [3.3440, -3.3620, -5.1101], [2.6017, -4.3613, -4.1444]]
        ).to(torch_device)
        torch.testing.assert_close(class_queries_logits[0, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        model = (
            OneFormerForUniversalSegmentation.from_pretrained(self.model_checkpoints)
            .to(torch_device, dtype=torch.float16)
            .eval()
        )
        processor = self.default_processor
        image = prepare_img()
        inputs = processor(image, ["semantic"], return_tensors="pt").to(torch_device, dtype=torch.float16)

        with torch.no_grad():
            _ = model(**inputs)

    def test_with_segmentation_maps_and_loss(self):
        dummy_model = OneFormerForUniversalSegmentation.from_pretrained(self.model_checkpoints)
        processor = self.default_processor
        processor.image_processor.num_text = dummy_model.config.num_queries - dummy_model.config.text_encoder_n_ctx
        dummy_model.config.is_training = True
        model = OneFormerForUniversalSegmentation(dummy_model.config).to(torch_device).eval()
        del dummy_model

        inputs = processor(
            [np.zeros((3, 512, 640)), np.zeros((3, 512, 640))],
            ["semantic", "semantic"],
            segmentation_maps=[np.zeros((384, 384)).astype(np.float32), np.zeros((384, 384)).astype(np.float32)],
            return_tensors="pt",
        )

        inputs["pixel_values"] = inputs["pixel_values"].to(torch_device)
        inputs["task_inputs"] = inputs["task_inputs"].to(torch_device)
        inputs["text_inputs"] = inputs["text_inputs"].to(torch_device)
        inputs["mask_labels"] = [el.to(torch_device) for el in inputs["mask_labels"]]
        inputs["class_labels"] = [el.to(torch_device) for el in inputs["class_labels"]]

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(outputs.loss is not None)
