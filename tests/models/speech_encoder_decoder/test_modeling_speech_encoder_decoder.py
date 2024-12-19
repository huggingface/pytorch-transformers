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


import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_deterministic_for_xpu,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from ..bert.test_modeling_bert import BertModelTester
from ..speech_to_text.test_modeling_speech_to_text import Speech2TextModelTester
from ..wav2vec2.test_modeling_wav2vec2 import Wav2Vec2ModelTester


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        BertLMHeadModel,
        SpeechEncoderDecoderConfig,
        SpeechEncoderDecoderModel,
        Wav2Vec2Model,
    )
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.speech_to_text.modeling_speech_to_text import Speech2TextEncoder


@require_torch
class EncoderDecoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_encoder_decoder_model_from_pretrained_configs(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_decoder_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = SpeechEncoderDecoderModel(encoder_decoder_config)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        self.assertFalse(enc_dec_model.config.tie_word_embeddings)

        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
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
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_with_inputs(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        inputs = input_values if input_features is None else input_features
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)

        outputs_encoder_decoder = enc_dec_model(
            inputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        outputs_encoder_decoder_kwarg = enc_dec_model(
            inputs=inputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder_kwarg["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_save_and_load(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                input_values=input_values,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                enc_dec_model.save_pretrained(tmpdirname)
                enc_dec_model = SpeechEncoderDecoderModel.from_pretrained(tmpdirname)
                enc_dec_model.to(torch_device)

                after_outputs = enc_dec_model(
                    input_values=input_values,
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_save_and_load_encoder_decoder_model(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                input_values=input_values,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:
                enc_dec_model.encoder.save_pretrained(encoder_tmp_dirname)
                enc_dec_model.decoder.save_pretrained(decoder_tmp_dirname)
                SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path=encoder_tmp_dirname,
                    decoder_pretrained_model_name_or_path=decoder_tmp_dirname,
                )

                after_outputs = enc_dec_model(
                    input_values=input_values,
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        inputs = input_values if input_features is None else input_features

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        seq_len = enc_dec_model.encoder._get_feat_extract_output_lengths(inputs.shape[1])
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

    def check_encoder_decoder_model_generate(
        self, config, decoder_config, input_values=None, input_features=None, **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)

        # make sure EOS token is set to None to prevent early stopping of generation
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None

        inputs = input_values if input_features is None else input_features

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

    def test_encoder_decoder_model_with_inputs(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_with_inputs(**input_ids_dict)

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

        model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        model.to(torch_device)
        model.train()
        model.gradient_checkpointing_enable()
        model.config.decoder_start_token_id = 0
        model.config.pad_token_id = 0

        model_inputs = {
            "attention_mask": inputs_dict["attention_mask"],
            "labels": inputs_dict["labels"],
            "decoder_input_ids": inputs_dict["decoder_input_ids"],
        }
        inputs = inputs_dict["input_features"] if "input_features" in inputs_dict else inputs_dict["input_values"]

        loss = model(inputs, **model_inputs).loss
        loss.backward()

    @slow
    @require_deterministic_for_xpu
    def test_real_model_save_load_from_pretrained(self):
        model_2, inputs = self.get_pretrained_model_and_inputs()
        model_2.to(torch_device)

        with torch.no_grad():
            outputs = model_2(**inputs)
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model_2.save_pretrained(tmp_dirname)
                model_1 = SpeechEncoderDecoderModel.from_pretrained(tmp_dirname)
                model_1.to(torch_device)

                after_outputs = model_1(**inputs)
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        inputs_dict = self.prepare_config_and_inputs()
        encoder_config, decoder_config = inputs_dict["config"], inputs_dict["decoder_config"]
        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config, decoder_config=decoder_config
        )
        model = SpeechEncoderDecoderModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model_sdpa = SpeechEncoderDecoderModel.from_pretrained(tmpdirname)
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
                model_sdpa_explicit = SpeechEncoderDecoderModel.from_pretrained(tmpdirname, attn_implementation="sdpa")
                model_sdpa_explicit = model_sdpa_explicit.eval().to(torch_device)

                self.assertTrue(model_sdpa_explicit.config._attn_implementation == "sdpa")
            else:
                with self.assertRaises(ValueError):
                    model_sdpa_explicit = SpeechEncoderDecoderModel.from_pretrained(
                        tmpdirname, attn_implementation="sdpa"
                    )

            model_eager = SpeechEncoderDecoderModel.from_pretrained(
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
class Wav2Vec2BertModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/wav2vec2-base-960h", "google-bert/bert-base-cased"
        )
        batch_size = 13
        input_values = floats_tensor([batch_size, 512], scale=1.0)
        attention_mask = random_attention_mask([batch_size, 512])
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = Wav2Vec2Model(config).eval()
        decoder_model = BertLMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(self)
        wav2vec2_model_tester = Wav2Vec2ModelTester(self)
        encoder_config_and_inputs = wav2vec2_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_values,
            input_mask,
        ) = encoder_config_and_inputs
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
            "input_values": input_values,
            "attention_mask": input_mask,
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
class Speech2TextBertModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/s2t-small-librispeech-asr", "google-bert/bert-base-cased"
        )
        batch_size = 13
        input_features = floats_tensor([batch_size, 7, 80], scale=1.0)
        attention_mask = random_attention_mask([batch_size, 7])
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = Speech2TextEncoder(config).eval()
        decoder_model = BertLMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(self)
        speech2text_model_tester = Speech2TextModelTester(self)
        encoder_config_and_inputs = speech2text_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()

        config, inputs = encoder_config_and_inputs
        input_features = inputs["input_features"]
        input_mask = inputs["attention_mask"]

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
            "input_features": input_features,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }

    @unittest.skip(reason="Cannot save full model as Speech2TextModel != Speech2TextEncoder")
    def test_encoder_decoder_model_from_pretrained_configs(self):
        pass

    @unittest.skip(reason="Cannot save full model as Speech2TextModel != Speech2TextEncoder")
    def test_save_and_load_from_pretrained(self):
        pass

    @require_deterministic_for_xpu
    @unittest.skip(reason="Cannot save full model as Speech2TextModel != Speech2TextEncoder")
    def test_real_model_save_load_from_pretrained(self):
        pass
