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
"""Testing suite for the PyTorch DPT model."""

import unittest

from transformers import DPTConfig
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DPTForDepthEstimation, DPTForSemanticSegmentation, DPTModel
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


if is_vision_available():
    from PIL import Image

    from transformers import DPTImageProcessor


class DPTModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=32,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        backbone_out_indices=[0, 1, 2, 3],
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        neck_hidden_sizes=[16, 32],
        is_hybrid=False,
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
        self.backbone_out_indices = backbone_out_indices
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.is_hybrid = is_hybrid
        self.neck_hidden_sizes = neck_hidden_sizes
        # sequence length of DPT = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DPTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            fusion_hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            backbone_out_indices=self.backbone_out_indices,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            is_hybrid=self.is_hybrid,
            neck_hidden_sizes=self.neck_hidden_sizes,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DPTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DPTForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DPTForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size, self.image_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DPTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as DPT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DPTModel, DPTForDepthEstimation, DPTForSemanticSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "depth-estimation": DPTForDepthEstimation,
            "image-feature-extraction": DPTModel,
            "image-segmentation": DPTForSemanticSegmentation,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DPTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="DPT does not use inputs_embeds")
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

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    def test_training(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DPTForDepthEstimation":
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
            if model_class.__name__ == "DPTForDepthEstimation":
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
                if module.__class__.__name__ == "DPTViTHybridEmbeddings":
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

    def test_backbone_selection(self):
        def _validate_backbone_init():
            for model_class in self.all_model_classes:
                model = model_class(config)
                model.to(torch_device)
                model.eval()

                if model.__class__.__name__ == "DPTForDepthEstimation":
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

    @slow
    def test_model_from_pretrained(self):
        model_name = "Intel/dpt-large"
        model = DPTModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DPTModelIntegrationTest(unittest.TestCase):
    def test_inference_depth_estimation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        expected_shape = torch.Size((1, 384, 384))
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_semantic_segmentation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large-ade")
        model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 150, 480, 480))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))

    def test_post_processing_semantic_segmentation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large-ade")
        model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        outputs.logits = outputs.logits.detach().cpu()

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[(500, 300)])
        expected_shape = torch.Size((500, 300))
        self.assertEqual(segmentation[0].shape, expected_shape)

        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = torch.Size((480, 480))
        self.assertEqual(segmentation[0].shape, expected_shape)

    def test_post_processing_depth_estimation(self):
        image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_depth = image_processor.post_process_depth_estimation(outputs=outputs)[0]["predicted_depth"]
        expected_shape = torch.Size((384, 384))
        self.assertTrue(predicted_depth.shape == expected_shape)

        predicted_depth_l = image_processor.post_process_depth_estimation(outputs=outputs, target_sizes=[(500, 500)])
        predicted_depth_l = predicted_depth_l[0]["predicted_depth"]
        expected_shape = torch.Size((500, 500))
        self.assertTrue(predicted_depth_l.shape == expected_shape)

        output_enlarged = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0).unsqueeze(1), size=(500, 500), mode="bicubic", align_corners=False
        ).squeeze()
        self.assertTrue(output_enlarged.shape == expected_shape)
        self.assertTrue(torch.allclose(predicted_depth_l, output_enlarged, rtol=1e-3))

    def test_export(self):
        for strict in [True, False]:
            with self.subTest(strict=strict):
                if not is_torch_greater_or_equal_than_2_4:
                    self.skipTest(reason="This test requires torch >= 2.4 to run.")
                model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade").to(torch_device).eval()
                image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large-ade")
                image = prepare_img()
                inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

                exported_program = torch.export.export(
                    model,
                    args=(inputs["pixel_values"],),
                    strict=strict,
                )
                with torch.no_grad():
                    eager_outputs = model(**inputs)
                    exported_outputs = exported_program.module().forward(inputs["pixel_values"])
                self.assertEqual(eager_outputs.logits.shape, exported_outputs.logits.shape)
                self.assertTrue(torch.allclose(eager_outputs.logits, exported_outputs.logits, atol=1e-4))
