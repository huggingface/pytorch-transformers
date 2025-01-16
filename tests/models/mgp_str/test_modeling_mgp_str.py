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
"""Testing suite for the PyTorch MGP-STR model."""

import unittest

import requests

from transformers import MgpstrConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import MgpstrForSceneTextRecognition, MgpstrModel


if is_vision_available():
    from PIL import Image

    from transformers import MgpstrProcessor


class MgpstrModelTester:
    def __init__(
        self,
        parent,
        is_training=False,
        batch_size=13,
        image_size=(32, 128),
        patch_size=4,
        num_channels=3,
        max_token_length=27,
        num_character_labels=38,
        num_bpe_labels=99,
        num_wordpiece_labels=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=4.0,
        patch_embeds_hidden_size=257,
        output_hidden_states=None,
    ):
        self.parent = parent
        self.is_training = is_training
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_token_length = max_token_length
        self.num_character_labels = num_character_labels
        self.num_bpe_labels = num_bpe_labels
        self.num_wordpiece_labels = num_wordpiece_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        self.output_hidden_states = output_hidden_states

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return MgpstrConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            max_token_length=self.max_token_length,
            num_character_labels=self.num_character_labels,
            num_bpe_labels=self.num_bpe_labels,
            num_wordpiece_labels=self.num_wordpiece_labels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            output_hidden_states=self.output_hidden_states,
        )

    def create_and_check_model(self, config, pixel_values):
        model = MgpstrForSceneTextRecognition(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            generated_ids = model(pixel_values)
        self.parent.assertEqual(
            generated_ids[0][0].shape, (self.batch_size, self.max_token_length, self.num_character_labels)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class MgpstrModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MgpstrForSceneTextRecognition,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": MgpstrForSceneTextRecognition, "image-feature-extraction": MgpstrModel}
        if is_torch_available()
        else {}
    )
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = MgpstrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MgpstrConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="MgpstrModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @unittest.skip(reason="MgpstrModel does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.patch_embeds_hidden_size, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # override as the `logit_scale` parameter initilization is different for MgpstrModel
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if isinstance(param, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                    if param.requires_grad:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass


# We will verify our results on an image from the IIIT-5k dataset
def prepare_img():
    url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return im


@require_vision
@require_torch
class MgpstrModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "alibaba-damo/mgp-str-base"
        model = MgpstrForSceneTextRecognition.from_pretrained(model_name).to(torch_device)
        processor = MgpstrProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(inputs)

        # verify the logits
        self.assertEqual(outputs.logits[0].shape, torch.Size((1, 27, 38)))

        out_strs = processor.batch_decode(outputs.logits)
        expected_text = "ticket"

        self.assertEqual(out_strs["generated_text"][0], expected_text)

        expected_slice = torch.tensor(
            [[[-39.5397, -44.4024, -36.1844], [-61.4709, -63.8639, -58.3454], [-74.0225, -68.5494, -71.2164]]],
            device=torch_device,
        )

        torch.testing.assert_close(outputs.logits[0][:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)
