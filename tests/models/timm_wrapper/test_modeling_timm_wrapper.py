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

import inspect
import tempfile
import unittest

from transformers.testing_utils import (
    require_bitsandbytes,
    require_timm,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_timm_available, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import TimmWrapperConfig, TimmWrapperForImageClassification, TimmWrapperModel


if is_timm_available():
    import timm


if is_vision_available():
    from PIL import Image

    from transformers import TimmWrapperImageProcessor


class TimmWrapperModelTester:
    def __init__(
        self,
        parent,
        model_name="timm/resnet18.a1_in1k",
        batch_size=3,
        image_size=32,
        num_channels=3,
        is_training=True,
    ):
        self.parent = parent
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return TimmWrapperConfig.from_pretrained(self.model_name)

    def create_and_check_model(self, config, pixel_values):
        model = TimmWrapperModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(
            result.feature_map[-1].shape,
            (self.batch_size, model.channels[-1], 14, 14),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
@require_timm
class TimmWrapperModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TimmWrapperModel, TimmWrapperForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": TimmWrapperModel, "image-classification": TimmWrapperForImageClassification}
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False
    has_attentions = False
    test_model_parallel = False

    def setUp(self):
        self.config_class = TimmWrapperConfig
        self.model_tester = TimmWrapperModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=self.config_class,
            has_text_modality=False,
            common_properties=[],
            model_name="timm/resnet18.a1_in1k",
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            # check all hidden states
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)
            self.assertTrue(
                len(outputs.hidden_states) == 5, f"expected 5 hidden states, but got {len(outputs.hidden_states)}"
            )
            expected_shapes = [[16, 16], [8, 8], [4, 4], [2, 2], [1, 1]]
            resulted_shapes = [list(h.shape[2:]) for h in outputs.hidden_states]
            self.assertListEqual(expected_shapes, resulted_shapes)

            # check we can select hidden states by indices
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=[-2, -1])
            self.assertTrue(
                len(outputs.hidden_states) == 2, f"expected 2 hidden states, but got {len(outputs.hidden_states)}"
            )
            expected_shapes = [[2, 2], [1, 1]]
            resulted_shapes = [list(h.shape[2:]) for h in outputs.hidden_states]
            self.assertListEqual(expected_shapes, resulted_shapes)

    @unittest.skip(reason="TimmWrapper models doesn't have inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TimmWrapper models doesn't have inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="TimmWrapper doesn't support output_attentions=True.")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="TimmWrapper doesn't support this.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="TimmWrapper initialization is managed on the timm side")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Need to use a timm model and there is no tiny model available.")
    def test_model_is_small(self):
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

    def test_do_pooling_option(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.do_pooling = False

        model = TimmWrapperModel._from_config(config)

        # check there is no pooling
        with torch.no_grad():
            output = model(**inputs_dict)
        self.assertIsNone(output.pooler_output)

        # check there is pooler output
        with torch.no_grad():
            output = model(**inputs_dict, do_pooling=True)
        self.assertIsNotNone(output.pooler_output)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_timm
@require_vision
class TimmWrapperModelIntegrationTest(unittest.TestCase):
    # some popular ones
    model_names_to_test = [
        "vit_small_patch16_384.augreg_in21k_ft_in1k",
        "resnet50.a1_in1k",
        "tf_mobilenetv3_large_minimal_100.in1k",
        "swin_tiny_patch4_window7_224.ms_in1k",
        "ese_vovnet19b_dw.ra_in1k",
        "hrnet_w18.ms_aug_in1k",
    ]

    @slow
    def test_inference_image_classification_head(self):
        checkpoint = "timm/resnet18.a1_in1k"
        model = TimmWrapperForImageClassification.from_pretrained(checkpoint, device_map=torch_device).eval()
        image_processor = TimmWrapperImageProcessor.from_pretrained(checkpoint)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the shape and logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_label = 281  # tabby cat
        self.assertEqual(torch.argmax(outputs.logits).item(), expected_label)

        expected_slice = torch.tensor([-11.2618, -9.6192, -10.3205]).to(torch_device)
        resulted_slice = outputs.logits[0, :3]
        is_close = torch.allclose(resulted_slice, expected_slice, atol=1e-3)
        self.assertTrue(is_close, f"Expected {expected_slice}, but got {resulted_slice}")

    @slow
    @require_bitsandbytes
    def test_inference_image_classification_quantized(self):
        from transformers import BitsAndBytesConfig

        checkpoint = "timm/vit_small_patch16_384.augreg_in21k_ft_in1k"

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = TimmWrapperForImageClassification.from_pretrained(
            checkpoint, quantization_config=quantization_config, device_map=torch_device
        ).eval()
        image_processor = TimmWrapperImageProcessor.from_pretrained(checkpoint)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the shape and logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_label = 281  # tabby cat
        self.assertEqual(torch.argmax(outputs.logits).item(), expected_label)

        expected_slice = torch.tensor([-2.4043, 1.4492, -0.5127]).to(outputs.logits.dtype)
        resulted_slice = outputs.logits[0, :3].cpu()
        is_close = torch.allclose(resulted_slice, expected_slice, atol=0.1)
        self.assertTrue(is_close, f"Expected {expected_slice}, but got {resulted_slice}")

    @slow
    def test_transformers_model_for_classification_is_equivalent_to_timm(self):
        # check that wrapper logits are the same as timm model logits

        image = prepare_img()

        for model_name in self.model_names_to_test:
            checkpoint = f"timm/{model_name}"

            with self.subTest(msg=model_name):
                # prepare inputs
                image_processor = TimmWrapperImageProcessor.from_pretrained(checkpoint)
                pixel_values = image_processor(images=image).pixel_values.to(torch_device)

                # load models
                model = TimmWrapperForImageClassification.from_pretrained(checkpoint, device_map=torch_device).eval()
                timm_model = timm.create_model(model_name, pretrained=True).to(torch_device).eval()

                with torch.inference_mode():
                    outputs = model(pixel_values)
                    timm_outputs = timm_model(pixel_values)

                # check shape is the same
                self.assertEqual(outputs.logits.shape, timm_outputs.shape)

                # check logits are the same
                diff = (outputs.logits - timm_outputs).max().item()
                self.assertLess(diff, 1e-4)

    @slow
    def test_transformers_model_is_equivalent_to_timm(self):
        # check that wrapper logits are the same as timm model logits

        image = prepare_img()

        models_to_test = ["vit_small_patch16_224.dino"] + self.model_names_to_test

        for model_name in models_to_test:
            checkpoint = f"timm/{model_name}"

            with self.subTest(msg=model_name):
                # prepare inputs
                image_processor = TimmWrapperImageProcessor.from_pretrained(checkpoint)
                pixel_values = image_processor(images=image).pixel_values.to(torch_device)

                # load models
                model = TimmWrapperModel.from_pretrained(checkpoint, device_map=torch_device).eval()
                timm_model = timm.create_model(model_name, pretrained=True, num_classes=0).to(torch_device).eval()

                with torch.inference_mode():
                    outputs = model(pixel_values)
                    timm_outputs = timm_model(pixel_values)

                # check shape is the same
                self.assertEqual(outputs.pooler_output.shape, timm_outputs.shape)

                # check logits are the same
                diff = (outputs.pooler_output - timm_outputs).max().item()
                self.assertLess(diff, 1e-4)

    @slow
    def test_save_load_to_timm(self):
        # test that timm model can be loaded to transformers, saved and then loaded back into timm

        model = TimmWrapperForImageClassification.from_pretrained(
            "timm/resnet18.a1_in1k", num_labels=10, ignore_mismatched_sizes=True
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            # there is no direct way to load timm model from folder, use the same config + path to weights
            timm_model = timm.create_model(
                "resnet18", num_classes=10, checkpoint_path=f"{tmpdirname}/model.safetensors"
            )

        # check that all weights are the same after reload
        different_weights = []
        for (name1, param1), (name2, param2) in zip(
            model.timm_model.named_parameters(), timm_model.named_parameters()
        ):
            if param1.shape != param2.shape or not torch.equal(param1, param2):
                different_weights.append((name1, name2))

        if different_weights:
            self.fail(f"Found different weights after reloading: {different_weights}")
