# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch VisionTextDualEncoder model."""

import collections
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import is_pt_flax_cross_test, require_torch, require_vision, slow, torch_device
from transformers.utils import is_flax_available, is_torch_available, is_vision_available

from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from ..bert.test_modeling_bert import BertModelTester
from ..clip.test_modeling_clip import CLIPVisionModelTester
from ..deit.test_modeling_deit import DeiTModelTester
from ..roberta.test_modeling_roberta import RobertaModelTester
from ..vit.test_modeling_vit import ViTModelTester


if is_torch_available():
    import torch

    from transformers import (
        BertModel,
        CLIPVisionModel,
        DeiTModel,
        RobertaModel,
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderModel,
        ViTModel,
    )

if is_flax_available():
    from transformers import FlaxVisionTextDualEncoderModel
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )

if is_vision_available():
    from PIL import Image

    from transformers import VisionTextDualEncoderProcessor


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


@require_torch
class VisionTextDualEncoderMixin:
    def get_vision_text_model(self, config, text_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_model_from_pretrained_configs(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)

        model = VisionTextDualEncoderModel(config)
        model.to(torch_device)
        model.eval()

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], config.projection_dim))

    def check_vision_text_dual_encoder_model(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_vision_text_dual_encoder_from_pretrained(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        kwargs = {"vision_model": vision_model, "text_model": text_model}
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(**kwargs)
        model.to(torch_device)
        model.eval()

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_save_load(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            out_1 = output[0].cpu().numpy()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = VisionTextDualEncoderModel.from_pretrained(tmpdirname).eval()
                model.to(torch_device)

                after_output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
                out_2 = after_output[0].cpu().numpy()
                max_diff = np.amax(np.abs(out_2 - out_1))
                self.assertLessEqual(max_diff, 1e-5)

    def check_vision_text_output_attention(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        output = model(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=True
        )

        vision_attentions = output.vision_model_output.attentions
        self.assertEqual(len(vision_attentions), vision_config.num_hidden_layers)

        # in ViT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(vision_model.config.image_size)
        patch_size = to_2tuple(vision_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        self.assertEqual(vision_attentions[0].shape[-3:], (vision_config.num_attention_heads, seq_len, seq_len))

        text_attentions = output.text_model_output.attentions
        self.assertEqual(len(text_attentions), text_config.num_hidden_layers)

        self.assertEqual(
            text_attentions[0].shape[-3:],
            (text_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]),
        )

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    def check_pt_flax_equivalence(self, pt_model, fx_model, input_ids, attention_mask, pixel_values, **kwargs):
        pt_model.to(torch_device)
        pt_model.eval()

        # prepare inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
        pt_inputs = inputs_dict
        flax_inputs = {k: v.numpy(force=True) for k, v in pt_inputs.items()}

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs).to_tuple()

        fx_outputs = fx_model(**flax_inputs).to_tuple()
        self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
        for fx_output, pt_output in zip(fx_outputs[:4], pt_outputs[:4]):
            self.assert_almost_equals(fx_output, pt_output.numpy(force=True), 4e-2)

        # PT -> Flax
        with tempfile.TemporaryDirectory() as tmpdirname:
            pt_model.save_pretrained(tmpdirname)
            fx_model_loaded = FlaxVisionTextDualEncoderModel.from_pretrained(tmpdirname, from_pt=True)

        fx_outputs_loaded = fx_model_loaded(**flax_inputs).to_tuple()
        self.assertEqual(len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
        for fx_output_loaded, pt_output in zip(fx_outputs_loaded[:4], pt_outputs[:4]):
            self.assert_almost_equals(fx_output_loaded, pt_output.numpy(force=True), 4e-2)

        # Flax -> PT
        with tempfile.TemporaryDirectory() as tmpdirname:
            fx_model.save_pretrained(tmpdirname)
            pt_model_loaded = VisionTextDualEncoderModel.from_pretrained(tmpdirname, from_flax=True)

        pt_model_loaded.to(torch_device)
        pt_model_loaded.eval()

        with torch.no_grad():
            pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

        self.assertEqual(len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch")
        for fx_output, pt_output_loaded in zip(fx_outputs[:4], pt_outputs_loaded[:4]):
            self.assert_almost_equals(fx_output, pt_output_loaded.numpy(force=True), 4e-2)

    def check_equivalence_pt_to_flax(self, vision_config, text_config, inputs_dict):
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)

        pt_model = VisionTextDualEncoderModel(config)
        fx_model = FlaxVisionTextDualEncoderModel(config)

        fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
        fx_model.params = fx_state

        self.check_pt_flax_equivalence(pt_model, fx_model, **inputs_dict)

    def check_equivalence_flax_to_pt(self, vision_config, text_config, inputs_dict):
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)

        pt_model = VisionTextDualEncoderModel(config)
        fx_model = FlaxVisionTextDualEncoderModel(config)

        pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

        self.check_pt_flax_equivalence(pt_model, fx_model, **inputs_dict)

    def test_vision_text_dual_encoder_model(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_dual_encoder_model(**inputs_dict)

    def test_model_from_pretrained_configs(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_model_from_pretrained_configs(**inputs_dict)

    def test_vision_text_dual_encoder_from_pretrained(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_dual_encoder_from_pretrained(**inputs_dict)

    def test_save_load(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_save_load(**inputs_dict)

    def test_vision_text_output_attention(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_output_attention(**inputs_dict)

    @is_pt_flax_cross_test
    def test_pt_flax_equivalence(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        vision_config = config_inputs_dict.pop("vision_config")
        text_config = config_inputs_dict.pop("text_config")

        inputs_dict = config_inputs_dict

        self.check_equivalence_pt_to_flax(vision_config, text_config, inputs_dict)
        self.check_equivalence_flax_to_pt(vision_config, text_config, inputs_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2, inputs = self.get_pretrained_model_and_inputs()
        model_2.to(torch_device)

        with torch.no_grad():
            outputs = model_2(**inputs)
            out_2 = outputs[0].cpu().numpy()

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model_2.save_pretrained(tmp_dirname)
                model_1 = VisionTextDualEncoderModel.from_pretrained(tmp_dirname)
                model_1.to(torch_device)

                after_outputs = model_1(**inputs)
                out_1 = after_outputs[0].cpu().numpy()
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)


@require_torch
class ViTBertModelTest(VisionTextDualEncoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            "hf-internal-testing/tiny-random-vit", "hf-internal-testing/tiny-bert"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.vision_model.config.num_channels,
                model.vision_model.config.image_size,
                model.vision_model.config.image_size,
            ]
        )
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return model, inputs

    def get_vision_text_model(self, vision_config, text_config):
        vision_model = ViTModel(vision_config).eval()
        text_model = BertModel(text_config).eval()
        return vision_model, text_model

    def prepare_config_and_inputs(self):
        vit_model_tester = ViTModelTester(self)
        bert_model_tester = BertModelTester(self)
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values, _ = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }


@require_torch
class DeiTRobertaModelTest(VisionTextDualEncoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            "hf-internal-testing/tiny-random-deit", "hf-internal-testing/tiny-random-roberta"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.vision_model.config.num_channels,
                model.vision_model.config.image_size,
                model.vision_model.config.image_size,
            ]
        )
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return model, inputs

    def check_vision_text_output_attention(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.to(torch_device)
        model.eval()

        output = model(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=True
        )

        vision_attentions = output.vision_model_output.attentions
        self.assertEqual(len(vision_attentions), vision_config.num_hidden_layers)

        # in DEiT, the seq_len equals the number of patches + 2 (we add 2 for the [CLS] and distillation tokens)
        image_size = to_2tuple(vision_model.config.image_size)
        patch_size = to_2tuple(vision_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 2
        self.assertEqual(vision_attentions[0].shape[-3:], (vision_config.num_attention_heads, seq_len, seq_len))

        text_attentions = output.text_model_output.attentions
        self.assertEqual(len(text_attentions), text_config.num_hidden_layers)

        self.assertEqual(
            text_attentions[0].shape[-3:],
            (text_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]),
        )

    def get_vision_text_model(self, vision_config, text_config):
        vision_model = DeiTModel(vision_config).eval()
        text_model = RobertaModel(text_config).eval()
        return vision_model, text_model

    def prepare_config_and_inputs(self):
        vit_model_tester = DeiTModelTester(self)
        bert_model_tester = RobertaModelTester(self)
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values, _ = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }

    @unittest.skip(reason="DeiT is not available in Flax")
    def test_pt_flax_equivalence(self):
        pass


@require_torch
class CLIPVisionBertModelTest(VisionTextDualEncoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            "hf-internal-testing/tiny-random-clip", "hf-internal-testing/tiny-bert"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.vision_model.config.num_channels,
                model.vision_model.config.image_size,
                model.vision_model.config.image_size,
            ]
        )
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return model, inputs

    def get_vision_text_model(self, vision_config, text_config):
        vision_model = CLIPVisionModel(vision_config).eval()
        text_model = BertModel(text_config).eval()
        return vision_model, text_model

    def prepare_config_and_inputs(self):
        clip_model_tester = CLIPVisionModelTester(self)
        bert_model_tester = BertModelTester(self)
        vision_config_and_inputs = clip_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }


@require_vision
@require_torch
class VisionTextDualEncoderIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian", logit_scale_init_value=1.0)
        processor = VisionTextDualEncoderProcessor.from_pretrained("clip-italian/clip-italian")

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(
            text=["una foto di un gatto", "una foto di un cane"], images=image, padding=True, return_tensors="pt"
        )

        outputs = model(**inputs)

        # verify the logits
        self.assertEqual(outputs.logits_per_image.shape, (inputs.pixel_values.shape[0], inputs.input_ids.shape[0]))
        self.assertEqual(
            outputs.logits_per_text.shape,
            (inputs.input_ids.shape[0], inputs.pixel_values.shape[0]),
        )

        expected_logits = torch.tensor([[1.2284727, 0.3104122]])

        torch.testing.assert_close(outputs.logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)
