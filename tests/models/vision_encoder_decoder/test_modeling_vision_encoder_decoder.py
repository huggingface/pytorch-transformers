# coding=utf-8
# Copyright 2021 HuggingFace Inc. team.
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

import re
import tempfile
import unittest

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from packaging import version

from transformers import DonutProcessor, NougatProcessor, TrOCRProcessor
from transformers.testing_utils import (
    require_levenshtein,
    require_nltk,
    require_sentencepiece,
    require_torch,
    require_torch_sdpa,
    require_vision,
    slow,
    to_2tuple,
    torch_device,
)
from transformers.utils import (
    cached_property,
    is_torch_available,
    is_vision_available,
)

from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from ..bart.test_modeling_bart import BartModelTester
from ..bert.test_modeling_bert import BertModelTester
from ..deit.test_modeling_deit import DeiTModelTester
from ..donut.test_modeling_donut_swin import DonutSwinModelTester
from ..gpt2.test_modeling_gpt2 import GPT2ModelTester
from ..layoutlmv3.test_modeling_layoutlmv3 import LayoutLMv3ModelTester
from ..swin.test_modeling_swin import SwinModelTester
from ..trocr.test_modeling_trocr import TrOCRStandaloneDecoderModelTester
from ..vit.test_modeling_vit import ViTModelTester


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        AutoTokenizer,
        BartForCausalLM,
        BertLMHeadModel,
        DeiTModel,
        DonutSwinModel,
        GPT2LMHeadModel,
        LayoutLMv3Model,
        SwinModel,
        TrOCRForCausalLM,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
        ViTModel,
    )
    from transformers.modeling_outputs import BaseModelOutput


if is_vision_available():
    import PIL
    from PIL import Image

    from transformers import ViTImageProcessor


@require_torch
class EncoderDecoderMixin:
    supports_sdpa = False

    def get_encoder_decoder_model(self, config, decoder_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_encoder_decoder_model_from_pretrained_configs(
        self, config, decoder_config, decoder_input_ids, decoder_attention_mask, pixel_values=None, **kwargs
    ):
        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = VisionEncoderDecoderModel(encoder_decoder_config)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model(
        self, config, decoder_config, decoder_input_ids, decoder_attention_mask, pixel_values=None, **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=outputs_encoder_decoder.encoder_hidden_states[-1])
        outputs_encoder_decoder = enc_dec_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        pixel_values=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_save_and_load(
        self, config, decoder_config, decoder_input_ids, decoder_attention_mask, pixel_values=None, **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                enc_dec_model.save_pretrained(tmpdirname)
                enc_dec_model = VisionEncoderDecoderModel.from_pretrained(tmpdirname)
                enc_dec_model.to(torch_device)

                after_outputs = enc_dec_model(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_save_and_load_encoder_decoder_model(
        self, config, decoder_config, decoder_input_ids, decoder_attention_mask, pixel_values=None, **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:
                enc_dec_model.encoder.save_pretrained(encoder_tmp_dirname)
                enc_dec_model.decoder.save_pretrained(decoder_tmp_dirname)
                VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path=encoder_tmp_dirname,
                    decoder_pretrained_model_name_or_path=decoder_tmp_dirname,
                )

                after_outputs = enc_dec_model(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        # in ViT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(encoder_model.config.image_size)
        patch_size = to_2tuple(encoder_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        self.assertEqual(encoder_attentions[0].shape[-3:], (config.num_attention_heads, seq_len, seq_len))

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),
        )

    def check_encoder_decoder_model_generate(self, config, decoder_config, pixel_values=None, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Generate until max length
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None
        enc_dec_model.to(torch_device)

        inputs = pixel_values

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            inputs,
            decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id,
            max_length=decoder_config.max_length,
        )
        self.assertEqual(generated_output.shape, (inputs.shape[0],) + (decoder_config.max_length,))

    def test_encoder_decoder_model(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained_configs(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained_configs(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=False)

    def test_encoder_decoder_model_from_pretrained_return_dict(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=True)

    def test_save_and_load_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load(**input_ids_dict)

    def test_save_and_load_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def test_training_gradient_checkpointing(self):
        inputs_dict = self.prepare_config_and_inputs()
        encoder_model, decoder_model = self.get_encoder_decoder_model(
            inputs_dict["config"], inputs_dict["decoder_config"]
        )

        model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        model.to(torch_device)
        model.train()
        model.gradient_checkpointing_enable()
        model.config.decoder_start_token_id = 0
        model.config.pad_token_id = 0

        model_inputs = {
            "pixel_values": inputs_dict["pixel_values"],
            "labels": inputs_dict["labels"],
            "decoder_input_ids": inputs_dict["decoder_input_ids"],
        }

        loss = model(**model_inputs).loss
        loss.backward()

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2, inputs = self.get_pretrained_model_and_inputs()
        model_2.to(torch_device)

        with torch.no_grad():
            outputs = model_2(**inputs)
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model_2.save_pretrained(tmp_dirname)
                model_1 = VisionEncoderDecoderModel.from_pretrained(tmp_dirname)
                model_1.to(torch_device)

                after_outputs = model_1(**inputs)
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        if not self.supports_sdpa:
            self.skipTest("SDPA is not supported")

        inputs_dict = self.prepare_config_and_inputs()
        encoder_config, decoder_config = inputs_dict["config"], inputs_dict["decoder_config"]
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config, decoder_config=decoder_config
        )
        model = VisionEncoderDecoderModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model_sdpa = VisionEncoderDecoderModel.from_pretrained(tmpdirname)
            model_sdpa = model_sdpa.eval().to(torch_device)

            # see https://github.com/huggingface/transformers/pull/32238
            # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
            encoder_attn = "sdpa" if model.encoder._supports_sdpa else "eager"
            decoder_attn = "sdpa" if model.decoder._supports_sdpa else "eager"
            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_sdpa.encoder.config._attn_implementation == encoder_attn)
            self.assertTrue(model_sdpa.decoder.config._attn_implementation == decoder_attn)

            # Also test that nothing break if we request SDPA explicitly, when both sub-parts support it.
            # If the model supports sdpa (i.e. all of sub-models supports it) we'll dispatch safely
            # Otherwise we should raise error that SDPA is not supported, as some of the sub-models doesn't support
            if encoder_attn == "sdpa" and decoder_attn == "sdpa":
                model_sdpa_explicit = VisionEncoderDecoderModel.from_pretrained(tmpdirname, attn_implementation="sdpa")
                model_sdpa_explicit = model_sdpa_explicit.eval().to(torch_device)

                self.assertTrue(model_sdpa_explicit.config._attn_implementation == "sdpa")
            else:
                with self.assertRaises(ValueError):
                    model_sdpa_explicit = VisionEncoderDecoderModel.from_pretrained(
                        tmpdirname, attn_implementation="sdpa"
                    )

            model_eager = VisionEncoderDecoderModel.from_pretrained(
                tmpdirname,
                attn_implementation="eager",
            )
            model_eager = model_eager.eval().to(torch_device)

            self.assertTrue(model_eager.config._attn_implementation == "eager")
            self.assertTrue(model_eager.encoder.config._attn_implementation == "eager")
            self.assertTrue(model_eager.decoder.config._attn_implementation == "eager")

            for name, submodule in model_eager.named_modules():
                class_name = submodule.__class__.__name__
                if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                    raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class DeiT2RobertaModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-deit", "hf-internal-testing/tiny-random-roberta"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.encoder.config.num_channels,
                model.encoder.config.image_size,
                model.encoder.config.image_size,
            ]
        )
        # for DEiT, the sequence length is equal to the number of patches + 2 (for the [CLS] and distillation tokens)
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        # in DEiT, the seq_len equals the number of patches + 2 (we add 2 for the [CLS] and distillation tokens)
        image_size = to_2tuple(encoder_model.config.image_size)
        patch_size = to_2tuple(encoder_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 2
        self.assertEqual(encoder_attentions[0].shape[-3:], (config.num_attention_heads, seq_len, seq_len))

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = DeiTModel(config).eval()
        decoder_model = BertLMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(self)
        deit_model_tester = DeiTModelTester(self)
        encoder_config_and_inputs = deit_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()
        config, pixel_values, _ = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_attention_mask,
            _,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }


@require_torch
class ViT2BertModelTest(EncoderDecoderMixin, unittest.TestCase):
    supports_sdpa = True  # one submodel support SDPA

    def get_pretrained_model_and_inputs(self):
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-vit", "hf-internal-testing/tiny-bert"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.encoder.config.num_channels,
                model.encoder.config.image_size,
                model.encoder.config.image_size,
            ]
        )
        # for ViT, the sequence length is equal to the number of patches + 1 (for the [CLS] token)
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = ViTModel(config).eval()
        decoder_model = BertLMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        vit_model_tester = ViTModelTester(self)
        bert_model_tester = BertModelTester(self)
        encoder_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()

        config, pixel_values, _ = encoder_config_and_inputs

        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_attention_mask,
            _,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }


@require_torch
class Swin2BartModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = SwinModel(config).eval()
        decoder_model = BartForCausalLM(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = SwinModelTester(self, batch_size=13, embed_dim=32)
        model_tester_decoder = BartModelTester(self, batch_size=13, hidden_size=32, max_position_embeddings=512)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs()
        config, pixel_values, _ = encoder_config_and_inputs
        decoder_config, decoder_inputs_dict = decoder_config_and_inputs
        decoder_inputs_dict["labels"] = decoder_inputs_dict["decoder_input_ids"]

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            **decoder_inputs_dict,
        }

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        # in Swin, the seq_len equals:
        seq_len = encoder_model.config.window_size**2
        self.assertEqual(encoder_attentions[0].shape[-3:], (config.num_attention_heads[0], seq_len, seq_len))

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        encoder_seq_len = ((config.image_size // config.patch_size) ** 2) // (4 ** (len(config.depths) - 1))
        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, encoder_seq_len),
        )

    @unittest.skip(reason="There are no published pretrained BART-causal checkpoints for now")
    def test_real_model_save_load_from_pretrained(self):
        pass


@require_torch
class ViT2TrOCR(EncoderDecoderMixin, unittest.TestCase):
    supports_sdpa = True  # one submodel support SDPA

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = ViTModel(config).eval()
        decoder_model = TrOCRForCausalLM(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = ViTModelTester(self, batch_size=13)
        model_tester_decoder = TrOCRStandaloneDecoderModelTester(
            self, batch_size=13, d_model=32, max_position_embeddings=512
        )
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs()
        config, pixel_values, _ = encoder_config_and_inputs
        (decoder_config, decoder_input_ids, decoder_attention_mask, _) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": decoder_input_ids,
        }

    @unittest.skip(reason="There are no published pretrained TrOCR checkpoints for now")
    def test_real_model_save_load_from_pretrained(self):
        pass


@require_torch
class LayoutLMv32TrOCR(EncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = LayoutLMv3Model(config).eval()
        decoder_model = TrOCRForCausalLM(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = LayoutLMv3ModelTester(self, batch_size=13, image_size=4, patch_size=2)
        model_tester_decoder = TrOCRStandaloneDecoderModelTester(
            self, batch_size=13, d_model=32, max_position_embeddings=512
        )
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = encoder_config_and_inputs
        (decoder_config, decoder_input_ids, decoder_attention_mask, _) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "bbox": bbox,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": decoder_input_ids,
        }

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_ids,
        pixel_values,
        labels=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            **kwargs,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        # LayoutLMv3's sequence length equals the number of text tokens + number of patches + 1 (we add 1 for the CLS token)
        text_seq_length = input_ids.shape[-1]
        image_seq_length = (encoder_model.config.input_size // encoder_model.config.patch_size) ** 2 + 1
        seq_len = text_seq_length + image_seq_length

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),
        )

    def check_encoder_decoder_model_generate(self, config, decoder_config, pixel_values=None, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Generate until max length
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None
        enc_dec_model.to(torch_device)

        generated_output = enc_dec_model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=enc_dec_model.config.decoder.bos_token_id,
            max_length=decoder_config.max_length,
            **kwargs,
        )
        self.assertEqual(generated_output.shape, (pixel_values.shape[0],) + (decoder_config.max_length,))

    @unittest.skip(reason="There are no published pretrained TrOCR checkpoints for now")
    def test_real_model_save_load_from_pretrained(self):
        pass


@require_torch
class VIT2GPT2Test(EncoderDecoderMixin, unittest.TestCase):
    supports_sdpa = True  # both submodels support SDPA

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = ViTModel(config).eval()
        decoder_model = GPT2LMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = ViTModelTester(self, batch_size=13)
        model_tester_decoder = GPT2ModelTester(self, batch_size=13, hidden_size=32, max_position_embeddings=512)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs()
        config, pixel_values, labels = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_head_mask,
            decoder_token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_head_mask": decoder_head_mask,
            "labels": decoder_input_ids,
        }

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        pixel_values,
        labels=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            **kwargs,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        seq_len = (encoder_model.config.image_size // encoder_model.config.patch_size) ** 2 + 1

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),  # 4 6 16
        )

    def check_encoder_decoder_model_generate(self, config, decoder_config, pixel_values=None, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Generate until max length
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None
        enc_dec_model.to(torch_device)

        generated_output = enc_dec_model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=enc_dec_model.config.decoder.bos_token_id,
            max_length=decoder_config.max_length,
            **kwargs,
        )
        self.assertEqual(generated_output.shape, (pixel_values.shape[0],) + (decoder_config.max_length,))

    @unittest.skip(reason="VIT2GPT2 also has an integration test for testinf save-load")
    def test_real_model_save_load_from_pretrained(self):
        pass


@require_torch
class Donut2GPT2Test(EncoderDecoderMixin, unittest.TestCase):
    supports_sdpa = True  # one submodel (GPT2) support SDPA

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = DonutSwinModel(config).eval()
        decoder_model = GPT2LMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = DonutSwinModelTester(self, batch_size=13)
        model_tester_decoder = GPT2ModelTester(self, batch_size=13, hidden_size=32, max_position_embeddings=512)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs()
        config, pixel_values, labels = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_head_mask,
            decoder_token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_head_mask": decoder_head_mask,
            "labels": decoder_input_ids,
        }

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        pixel_values,
        labels=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            **kwargs,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        seq_len = encoder_model.config.image_size // encoder_model.config.patch_size

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),  # 4 6 16
        )

    def check_encoder_decoder_model_generate(self, config, decoder_config, pixel_values=None, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Generate until max length
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None
        enc_dec_model.to(torch_device)

        generated_output = enc_dec_model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=enc_dec_model.config.decoder.bos_token_id,
            max_length=decoder_config.max_length,
            **kwargs,
        )
        self.assertEqual(generated_output.shape, (pixel_values.shape[0],) + (decoder_config.max_length,))

    @unittest.skip(reason="Donut has an Integration test for that")
    def test_real_model_save_load_from_pretrained(self):
        pass


@require_vision
@require_torch
class TrOCRModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") if is_vision_available() else None

    @slow
    def test_inference_handwritten(self):
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(torch_device)

        dataset = load_dataset("hf-internal-testing/fixtures_ocr", split="test", trust_remote_code=True)
        image = Image.open(dataset[0]["file"]).convert("RGB")

        processor = self.default_processor
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # forward pass
        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(torch_device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [-1.4502, -4.6683, -0.5347, -2.9291, 9.1435, -3.0571, 8.9764, 1.7560, 8.7358, -1.5311]
        ).to(torch_device)

        torch.testing.assert_close(logits[0, 0, :10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_printed(self):
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(torch_device)

        dataset = load_dataset("hf-internal-testing/fixtures_ocr", split="test", trust_remote_code=True)
        image = Image.open(dataset[1]["file"]).convert("RGB")

        processor = self.default_processor
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # forward pass
        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(torch_device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        is_pillow_less_than_9 = version.parse(PIL.__version__) < version.parse("9.0.0")

        if is_pillow_less_than_9:
            expected_slice = torch.tensor(
                [-5.6816, -5.8388, 1.1398, -6.9034, 6.8505, -2.4393, 1.2284, -1.0232, -1.9661, -3.9210],
                device=torch_device,
            )
        else:
            expected_slice = torch.tensor(
                [-5.6844, -5.8372, 1.1518, -6.8984, 6.8587, -2.4453, 1.2347, -1.0241, -1.9649, -3.9109],
                device=torch_device,
            )

        torch.testing.assert_close(logits[0, 0, :10], expected_slice, rtol=1e-4, atol=1e-4)


@require_vision
@require_torch
class ViT2GPT2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_coco_en(self):
        loc = "ydshieh/vit-gpt2-coco-en"

        image_processor = ViTImageProcessor.from_pretrained(loc)
        tokenizer = AutoTokenizer.from_pretrained(loc)
        model = VisionEncoderDecoderModel.from_pretrained(loc)
        model.to(torch_device)
        model.eval()

        # We will verify our results on an image of cute cats
        img = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(torch_device)

        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(torch_device)

        with torch.no_grad():
            logits = model(pixel_values, decoder_input_ids)[0].detach().cpu().numpy()

        # verify the logits
        expected_shape = (1, 1, model.config.decoder.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

        EXPECTED_LOGIT_SLICE = np.array(
            [
                -38.705807,
                -30.639929,
                -31.41903,
                -39.012012,
                -38.38696,
                -34.887207,
                -33.290855,
                -35.68447,
                -38.508484,
                -36.124645,
            ]
        )
        max_diff = np.amax(np.abs(logits[0, 0, :10] - EXPECTED_LOGIT_SLICE))
        self.assertLessEqual(max_diff, 1e-4)

        def generate_step(pixel_values):
            outputs = model.generate(
                pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True, output_scores=True
            )
            output_ids = outputs.sequences
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            return preds, outputs.sequences_scores.detach().cpu().numpy()

        preds, scores = generate_step(pixel_values)

        EXPECTED_SCORES = np.array([-0.5956343])
        max_diff = np.amax(np.abs(scores - EXPECTED_SCORES))
        self.assertLessEqual(max_diff, 1e-4)

        # should produce
        # ["a cat laying on top of a couch next to another cat"]
        self.assertEqual(preds, ["a cat laying on top of a couch next to another cat"])


@require_vision
@require_torch
@require_sentencepiece
class DonutModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_docvqa(self):
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa").to(
            torch_device
        )

        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        image = dataset[0]["image"]

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)
        decoder_input_ids = processor.tokenizer(
            "<s_docvqa>", add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(torch_device)

        # step 1: single forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size([1, 1, 57532])
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([24.3873, -6.4491, 32.5394]).to(torch_device)
        torch.testing.assert_close(logits[0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # step 2: generation
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        question = "When is the coffee break?"
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(torch_device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            output_scores=True,
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

        # verify generated sequence
        self.assertEqual(
            sequence, "<s_question> When is the coffee break?</s_question><s_answer> 11-14 to 11:39 a.m.</s_answer>"
        )

        # verify scores
        self.assertEqual(len(outputs.scores), 11)
        self.assertTrue(
            torch.allclose(
                outputs.scores[0][0, :3], torch.tensor([5.6019, -3.5070, 13.7123], device=torch_device), atol=1e-4
            )
        )

    @slow
    def test_inference_cordv2(self):
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").to(
            torch_device
        )

        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        image = dataset[2]["image"]

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)
        decoder_input_ids = processor.tokenizer(
            "<s_cord-v2>", add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(torch_device)

        # step 1: single forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-27.4344, -3.2686, -19.3524], device=torch_device)
        torch.testing.assert_close(logits[0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # step 2: generation
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(torch_device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            output_scores=True,
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

        # verify generated sequence
        expected_sequence = "<s_menu><s_nm> CINNAMON SUGAR</s_nm><s_unitprice> 17,000</s_unitprice><s_cnt> 1 x</s_cnt><s_price> 17,000</s_price></s_menu><s_sub_total><s_subtotal_price> 17,000</s_subtotal_price></s_sub_total><s_total><s_total_price> 17,000</s_total_price><s_cashprice> 20,000</s_cashprice><s_changeprice> 3,000</s_changeprice></s_total>"  # noqa: E231  # fmt: skip
        self.assertEqual(sequence, expected_sequence)

        # verify scores
        self.assertEqual(len(outputs.scores), 43)
        self.assertTrue(
            torch.allclose(
                outputs.scores[0][0, :3], torch.tensor([-27.4344, -3.2686, -19.3524], device=torch_device), atol=1e-4
            )
        )

    @slow
    def test_inference_rvlcdip(self):
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip").to(
            torch_device
        )

        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        image = dataset[1]["image"]

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # step 1: single forward pass
        decoder_input_ids = processor.tokenizer(
            "<s_rvlcdip>", add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(torch_device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-17.6490, -4.8381, -15.7577], device=torch_device)
        torch.testing.assert_close(logits[0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # step 2: generation
        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(torch_device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            output_scores=True,
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

        # verify generated sequence
        self.assertEqual(sequence, "<s_class><advertisement/></s_class>")

        # verify scores
        self.assertEqual(len(outputs.scores), 4)
        self.assertTrue(
            torch.allclose(
                outputs.scores[0][0, :3], torch.tensor([-17.6490, -4.8381, -15.7577], device=torch_device), atol=1e-4
            )
        )


@require_levenshtein
@require_nltk
@require_torch
@require_vision
@slow
class NougatModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return NougatProcessor.from_pretrained("facebook/nougat-base") if is_vision_available() else None

    @cached_property
    def default_model(self):
        return VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base").to(torch_device)

    @cached_property
    def default_image(self):
        filepath = hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_pdf.png", repo_type="dataset"
        )
        image = Image.open(filepath).convert("RGB")
        return image

    def test_forward_pass(self):
        processor = self.default_processor
        model = self.default_model
        image = self.default_image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        decoder_input_ids = torch.tensor([[0]]).to(torch_device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [1.6253, -4.2179, 5.8532, -2.7911, -5.0609, -4.7397, -4.2890, -5.1073, -4.8908, -4.9729]
        ).to(torch_device)

        torch.testing.assert_close(logits[0, 0, :10], expected_slice, rtol=1e-4, atol=1e-4)

    def test_generation(self):
        processor = self.default_processor
        model = self.default_model
        image = self.default_image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        outputs = model.generate(
            pixel_values,
            min_length=1,
            max_length=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )

        # verify generated sequence
        generated = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        expected_raw_generation = "# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@meta.com\n\nGuillem Cucurull\n\nThomas Scialom\n\nRobert Stojnic\n\nMeta AI\n\nThe paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\n###### Abstract\n\nScientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (**N**eural **O**ptical **U**nderstanding for **A**cademic Documents), a Visual Transformer model that performs an _Optical Character Recognition_ (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.\n\n## 1 Introduction\n\nThe majority of scientific knowledge is stored in books or published in scientific journals, most commonly in the Portable Document Format (PDF). Next to HTML, PDFs are the second most prominent data format on the internet, making up 2.4% of common crawl [1]. However, the information stored in these files is very difficult to extract into any other formats. This is especially true for highly specialized documents, such as scientific research papers, where the semantic information of mathematical expressions is lost.\n\nExisting Optical Character Recognition (OCR) engines, such as Tesseract OCR [2], excel at detecting and classifying individual characters and words in an image, but fail to understand the relationship between them due to their line-by-line approach. This means that they treat superscripts and subscripts in the same way as the surrounding text, which is a significant drawback for mathematical expressions. In mathematical notations like fractions, exponents, and matrices, relative positions of characters are crucial.\n\nConverting academic research papers into machine-readable text also enables accessibility and searchability of science as a whole. The information of millions of academic papers can not be fully accessed because they are locked behind an unreadable format. Existing corpora, such as the S2ORC dataset [3], capture the text of 12M2 papers using GROBID [4], but are missing meaningful representations of the mathematical equations.\n\nFootnote 2: The paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\nTo this end, we introduce Nougat, a transformer based model that can convert images of document pages to formatted markup text.\n\nThe primary contributions in this paper are\n\n* Release of a pre-trained model capable of converting a PDF to a lightweight markup language. We release the code and the model on GitHub3 Footnote 3: https://github.com/facebookresearch/nougat\n* We introduce a pipeline to create dataset for pairing PDFs to source code\n* Our method is only dependent on the image of a page, allowing access to scanned papers and books"
        self.assertTrue(generated == expected_raw_generation)

        # verify postprocessed sequence
        generated = processor.post_process_generation(generated, fix_markdown=False)
        expected_generation = "\n\n# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@meta.com\n\nGuillem Cucurull\n\nThomas Scialom\n\nRobert Stojnic\n\nMeta AI\n\nThe paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\n###### Abstract\n\nScientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (**N**eural **O**ptical **U**nderstanding for **A**cademic Documents), a Visual Transformer model that performs an _Optical Character Recognition_ (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.\n\n## 1 Introduction\n\nThe majority of scientific knowledge is stored in books or published in scientific journals, most commonly in the Portable Document Format (PDF). Next to HTML, PDFs are the second most prominent data format on the internet, making up 2.4% of common crawl [1]. However, the information stored in these files is very difficult to extract into any other formats. This is especially true for highly specialized documents, such as scientific research papers, where the semantic information of mathematical expressions is lost.\n\nExisting Optical Character Recognition (OCR) engines, such as Tesseract OCR [2], excel at detecting and classifying individual characters and words in an image, but fail to understand the relationship between them due to their line-by-line approach. This means that they treat superscripts and subscripts in the same way as the surrounding text, which is a significant drawback for mathematical expressions. In mathematical notations like fractions, exponents, and matrices, relative positions of characters are crucial.\n\nConverting academic research papers into machine-readable text also enables accessibility and searchability of science as a whole. The information of millions of academic papers can not be fully accessed because they are locked behind an unreadable format. Existing corpora, such as the S2ORC dataset [3], capture the text of 12M2 papers using GROBID [4], but are missing meaningful representations of the mathematical equations.\n\nFootnote 2: The paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\nTo this end, we introduce Nougat, a transformer based model that can convert images of document pages to formatted markup text.\n\nThe primary contributions in this paper are\n\n* Release of a pre-trained model capable of converting a PDF to a lightweight markup language. We release the code and the model on GitHub3 Footnote 3: https://github.com/facebookresearch/nougat\n* We introduce a pipeline to create dataset for pairing PDFs to source code\n* Our method is only dependent on the image of a page, allowing access to scanned papers and books"
        self.assertTrue(generated == expected_generation)

        # verify scores
        self.assertEqual(len(outputs.scores), 741)
        self.assertTrue(
            torch.allclose(
                outputs.scores[0][0, :3], torch.tensor([1.6253, -4.2179, 5.8532], device=torch_device), atol=1e-4
            )
        )
