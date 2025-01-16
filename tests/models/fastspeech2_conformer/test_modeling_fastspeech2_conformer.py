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
"""Testing suite for the PyTorch FastSpeech2Conformer model."""

import inspect
import tempfile
import unittest

from transformers import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerTokenizer,
    FastSpeech2ConformerWithHifiGanConfig,
    is_torch_available,
)
from transformers.testing_utils import require_g2p_en, require_torch, require_torch_accelerator, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor


if is_torch_available():
    import torch

    from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerWithHifiGan, set_seed


class FastSpeech2ConformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=24,
        seq_length=7,
        encoder_linear_units=384,
        decoder_linear_units=384,
        is_training=False,
        speech_decoder_postnet_units=128,
        speech_decoder_postnet_layers=2,
        pitch_predictor_layers=1,
        energy_predictor_layers=1,
        duration_predictor_layers=1,
        num_mel_bins=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.encoder_linear_units = encoder_linear_units
        self.decoder_linear_units = decoder_linear_units
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.pitch_predictor_layers = pitch_predictor_layers
        self.energy_predictor_layers = energy_predictor_layers
        self.duration_predictor_layers = duration_predictor_layers
        self.num_mel_bins = num_mel_bins

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids

    def get_config(self):
        return FastSpeech2ConformerConfig(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_linear_units=self.encoder_linear_units,
            decoder_linear_units=self.decoder_linear_units,
            speech_decoder_postnet_units=self.speech_decoder_postnet_units,
            speech_decoder_postnet_layers=self.speech_decoder_postnet_layers,
            num_mel_bins=self.num_mel_bins,
            pitch_predictor_layers=self.pitch_predictor_layers,
            energy_predictor_layers=self.energy_predictor_layers,
            duration_predictor_layers=self.duration_predictor_layers,
        )

    def create_and_check_model(self, config, input_ids, *args):
        model = FastSpeech2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, return_dict=True)

        # total of 5 keys in result
        self.parent.assertEqual(len(result), 5)
        # check batch sizes match
        for value in result.values():
            self.parent.assertEqual(value.size(0), self.batch_size)
        # check duration, pitch, and energy have the appopriate shapes
        # duration: (batch_size, max_text_length), pitch and energy: (batch_size, max_text_length, 1)
        self.parent.assertEqual(result["duration_outputs"].shape + (1,), result["pitch_outputs"].shape)
        self.parent.assertEqual(result["pitch_outputs"].shape, result["energy_outputs"].shape)
        # check predicted mel-spectrogram has correct dimension
        self.parent.assertEqual(result["spectrogram"].size(2), model.config.num_mel_bins)

    def prepare_config_and_inputs_for_common(self):
        config, input_ids = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch_accelerator
@require_torch
class FastSpeech2ConformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FastSpeech2ConformerModel,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_resize_embeddings = False
    is_encoder_decoder = True

    def setUp(self):
        self.model_tester = FastSpeech2ConformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FastSpeech2ConformerConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    msg = f"Parameter {name} of model {model_class} seems not properly initialized"
                    if "norm" in name:
                        if "bias" in name:
                            self.assertEqual(param.data.mean().item(), 0.0, msg=msg)
                        if "weight" in name:
                            self.assertEqual(param.data.mean().item(), 1.0, msg=msg)
                    elif "conv" in name or "embed" in name:
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0, msg=msg)

    def test_duration_energy_pitch_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = self.model_tester.seq_length
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # duration
            self.assertListEqual(list(outputs.duration_outputs.shape), [self.model_tester.batch_size, seq_len])
            # energy
            self.assertListEqual(list(outputs.energy_outputs.shape), [self.model_tester.batch_size, seq_len, 1])
            # pitch
            self.assertListEqual(list(outputs.pitch_outputs.shape), [self.model_tester.batch_size, seq_len, 1])

    def test_hidden_states_output(self):
        def _check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            for idx, hidden_states in enumerate([outputs.encoder_hidden_states, outputs.decoder_hidden_states]):
                expected_num_layers = getattr(
                    self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
                )

                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertIsInstance(hidden_states, (list, tuple))
                expected_batch_size, expected_seq_length, expected_hidden_size = hidden_states[0].shape
                self.assertEqual(expected_batch_size, self.model_tester.batch_size)
                # Only test encoder seq_length since decoder seq_length is variable based on inputs
                if idx == 0:
                    self.assertEqual(expected_seq_length, self.model_tester.seq_length)
                self.assertEqual(expected_hidden_size, self.model_tester.hidden_size)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict["output_hidden_states"] = True
        _check_hidden_states_output(inputs_dict, config, FastSpeech2ConformerModel)

        # check that output_hidden_states also work using config
        del inputs_dict["output_hidden_states"]
        config.output_hidden_states = True

        _check_hidden_states_output(inputs_dict, config, FastSpeech2ConformerModel)

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = FastSpeech2ConformerModel(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            _, info = FastSpeech2ConformerModel.from_pretrained(tmpdirname, output_loading_info=True)
        self.assertEqual(info["missing_keys"], [])

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = FastSpeech2ConformerModel(config)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = [
            "input_ids",
            "attention_mask",
            "spectrogram_labels",
            "duration_labels",
            "pitch_labels",
            "energy_labels",
            "speaker_ids",
            "lang_ids",
            "speaker_embedding",
            "return_dict",
            "output_attentions",
            "output_hidden_states",
        ]
        self.assertListEqual(arg_names, expected_arg_names)

    # Override as FastSpeech2Conformer does not output cross attentions
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        model = FastSpeech2ConformerModel(config)
        model.to(torch_device)
        model.eval()

        inputs = self._prepare_for_class(inputs_dict, FastSpeech2ConformerModel)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        decoder_hidden_states = outputs.decoder_hidden_states[0]
        decoder_hidden_states.retain_grad()

        encoder_attentions = outputs.encoder_attentions[0]
        encoder_attentions.retain_grad()

        decoder_attentions = outputs.decoder_attentions[0]
        decoder_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(decoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(decoder_attentions.grad)

    def test_attention_outputs(self):
        """
        Custom `test_attention_outputs` since FastSpeech2Conformer does not output cross attentions, has variable
        decoder attention shape, and uniquely outputs energy, pitch, and durations.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = self.model_tester.seq_length

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.encoder_attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            encoder_attentions = outputs.encoder_attentions
            self.assertEqual(len(encoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(encoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            correct_outlen = 7
            self.assertEqual(out_len, correct_outlen)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )

    @slow
    def test_model_from_pretrained(self):
        model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        self.assertIsNotNone(model)

    @unittest.skip(reason="FastSpeech2Conformer does not accept inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="FastSpeech2Conformer has no input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        "FastSpeech2Conformer predicts durations in linear domain during inference"
        "Even small differences on hidden states lead to different durations, due to `torch.round`"
    )
    def test_batching_equivalence(self):
        pass


@require_torch
@require_g2p_en
@slow
class FastSpeech2ConformerModelIntegrationTest(unittest.TestCase):
    def test_inference_integration(self):
        model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        model.to(torch_device)
        model.eval()

        tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        text = "Test that this generates speech"
        input_ids = tokenizer(text, return_tensors="pt").to(torch_device)["input_ids"]

        outputs_dict = model(input_ids)
        spectrogram = outputs_dict["spectrogram"]

        # mel-spectrogram is too large (1, 205, 80), so only check top-left 100 elements
        # fmt: off
        expected_mel_spectrogram = torch.tensor(
            [
                [-1.2426, -1.7286, -1.6754, -1.7451, -1.6402, -1.5219, -1.4480, -1.3345, -1.4031, -1.4497],
                [-0.7858, -1.4966, -1.3602, -1.4876, -1.2949, -1.0723, -1.0021, -0.7553, -0.6521, -0.6929],
                [-0.7298, -1.3908, -1.0369, -1.2656, -1.0342, -0.7883, -0.7420, -0.5249, -0.3734, -0.3977],
                [-0.4784, -1.3508, -1.1558, -1.4678, -1.2820, -1.0252, -1.0868, -0.9006, -0.8947, -0.8448],
                [-0.3963, -1.2895, -1.2813, -1.6147, -1.4658, -1.2560, -1.4134, -1.2650, -1.3255, -1.1715],
                [-1.4914, -1.3097, -0.3821, -0.3898, -0.5748, -0.9040, -1.0755, -1.0575, -1.2205, -1.0572],
                [0.0197, -0.0582, 0.9147, 1.1512, 1.1651, 0.6628, -0.1010, -0.3085, -0.2285, 0.2650],
                [1.1780, 0.1803, 0.7251, 1.5728, 1.6678, 0.4542, -0.1572, -0.1787, 0.0744, 0.8168],
                [-0.2078, -0.3211, 1.1096, 1.5085, 1.4632, 0.6299, -0.0515, 0.0589, 0.8609, 1.4429],
                [0.7831, -0.2663, 1.0352, 1.4489, 0.9088, 0.0247, -0.3995, 0.0078, 1.2446, 1.6998],
            ],
            device=torch_device,
        )
        # fmt: on

        torch.testing.assert_close(spectrogram[0, :10, :10], expected_mel_spectrogram, rtol=1e-4, atol=1e-4)
        self.assertEqual(spectrogram.shape, (1, 205, model.config.num_mel_bins))

    def test_training_integration(self):
        model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        model.to(torch_device)
        # Set self.training manually to keep deterministic but run the training path
        model.training = True
        set_seed(0)

        tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        text = "Test that this generates speech"
        input_ids = tokenizer(text, return_tensors="pt").to(torch_device)["input_ids"]

        # NOTE: Dummy numbers since FastSpeech2Conformer does not have a feature extractor due to the package deps required (librosa, MFA)
        batch_size, max_text_len = input_ids.shape
        pitch_labels = torch.rand((batch_size, max_text_len, 1), dtype=torch.float, device=torch_device)
        energy_labels = torch.rand((batch_size, max_text_len, 1), dtype=torch.float, device=torch_device)
        duration_labels = torch.normal(10, 2, size=(batch_size, max_text_len)).clamp(1, 20).int()
        max_target_len, _ = duration_labels.sum(dim=1).max(dim=0)
        max_target_len = max_target_len.item()
        spectrogram_labels = torch.rand(
            (batch_size, max_target_len, model.num_mel_bins), dtype=torch.float, device=torch_device
        )

        outputs_dict = model(
            input_ids,
            spectrogram_labels=spectrogram_labels,
            duration_labels=duration_labels,
            pitch_labels=pitch_labels,
            energy_labels=energy_labels,
            return_dict=True,
        )
        spectrogram = outputs_dict["spectrogram"]
        loss = outputs_dict["loss"]

        # # mel-spectrogram is too large (1, 224, 80), so only check top-left 100 elements
        # fmt: off
        expected_mel_spectrogram = torch.tensor(
            [
                [-1.0643e+00, -6.8058e-01, -1.0901e+00, -8.2724e-01, -7.7241e-01, -1.1905e+00, -8.5725e-01, -8.2930e-01, -1.1313e+00, -1.2449e+00],
                [-5.5067e-01, -2.7045e-01, -6.3483e-01, -1.9320e-01,  1.0234e-01, -3.3253e-01, -2.4423e-01, -3.5045e-01, -5.2070e-01, -4.3710e-01],
                [ 2.2181e-01,  3.1433e-01, -1.2849e-01,  6.0253e-01,  1.0033e+00, 1.3952e-01,  1.2851e-01, -2.3063e-02, -1.5092e-01,  2.4903e-01],
                [ 4.6343e-01,  4.1820e-01,  1.6468e-01,  1.1297e+00,  1.4588e+00, 1.3737e-01,  6.6355e-02, -6.0973e-02, -5.4225e-02,  5.9208e-01],
                [ 5.2762e-01,  4.8725e-01,  4.2735e-01,  1.4392e+00,  1.7398e+00, 2.4891e-01, -8.4531e-03, -8.1282e-02,  1.2857e-01,  8.7559e-01],
                [ 5.2548e-01,  5.1653e-01,  5.2034e-01,  1.3782e+00,  1.5972e+00, 1.6380e-01, -5.1807e-02,  1.5474e-03,  2.2824e-01,  8.5288e-01],
                [ 3.6356e-01,  4.4109e-01,  4.4257e-01,  9.4273e-01,  1.1201e+00, -9.0551e-03, -1.1627e-01, -2.0821e-02,  1.0793e-01,  5.0336e-01],
                [ 3.6598e-01,  3.2708e-01,  1.3297e-01,  4.5162e-01,  6.4168e-01, -2.6923e-01, -2.3101e-01, -1.4943e-01, -1.4732e-01,  7.3057e-02],
                [ 2.7639e-01,  2.2588e-01, -1.5310e-01,  1.0957e-01,  3.3048e-01, -5.3431e-01, -3.3822e-01, -2.8007e-01, -3.3823e-01, -1.5775e-01],
                [ 2.9323e-01,  1.6723e-01, -3.4153e-01, -1.1209e-01,  1.7355e-01, -6.1724e-01, -5.4201e-01, -4.9944e-01, -5.2212e-01, -2.7596e-01]
            ],
            device=torch_device,
        )
        # fmt: on

        expected_loss = torch.tensor(74.4595, device=torch_device)

        torch.testing.assert_close(spectrogram[0, :10, :10], expected_mel_spectrogram, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-4)
        self.assertEqual(spectrogram.shape, (1, 224, model.config.num_mel_bins))


class FastSpeech2ConformerWithHifiGanTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=24,
        seq_length=7,
        encoder_linear_units=384,
        decoder_linear_units=384,
        is_training=False,
        speech_decoder_postnet_units=128,
        speech_decoder_postnet_layers=2,
        pitch_predictor_layers=1,
        energy_predictor_layers=1,
        duration_predictor_layers=1,
        num_mel_bins=8,
        upsample_initial_channel=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.encoder_linear_units = encoder_linear_units
        self.decoder_linear_units = decoder_linear_units
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.pitch_predictor_layers = pitch_predictor_layers
        self.energy_predictor_layers = energy_predictor_layers
        self.duration_predictor_layers = duration_predictor_layers
        self.num_mel_bins = num_mel_bins
        self.upsample_initial_channel = upsample_initial_channel

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids

    def get_config(self):
        self.model_config = FastSpeech2ConformerConfig(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_linear_units=self.encoder_linear_units,
            decoder_linear_units=self.decoder_linear_units,
            speech_decoder_postnet_units=self.speech_decoder_postnet_units,
            speech_decoder_postnet_layers=self.speech_decoder_postnet_layers,
            num_mel_bins=self.num_mel_bins,
            pitch_predictor_layers=self.pitch_predictor_layers,
            energy_predictor_layers=self.energy_predictor_layers,
            duration_predictor_layers=self.duration_predictor_layers,
        )
        self.vocoder_config = FastSpeech2ConformerHifiGanConfig(
            model_in_dim=self.num_mel_bins, upsample_initial_channel=self.upsample_initial_channel
        )
        return FastSpeech2ConformerWithHifiGanConfig(
            model_config=self.model_config.to_dict(), vocoder_config=self.vocoder_config.to_dict()
        )

    def create_and_check_model(self, config, input_ids, *args):
        model = FastSpeech2ConformerWithHifiGan(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, return_dict=True)

        # total of 5 keys in result
        self.parent.assertEqual(len(result), 6)
        # check batch sizes match
        for value in result.values():
            self.parent.assertEqual(value.size(0), self.batch_size)
        # check duration, pitch, and energy have the appopriate shapes
        # duration: (batch_size, max_text_length), pitch and energy: (batch_size, max_text_length, 1)
        self.parent.assertEqual(result["duration_outputs"].shape + (1,), result["pitch_outputs"].shape)
        self.parent.assertEqual(result["pitch_outputs"].shape, result["energy_outputs"].shape)
        # check predicted mel-spectrogram has correct dimension
        self.parent.assertEqual(result["spectrogram"].size(2), model.config.model_config.num_mel_bins)

    def prepare_config_and_inputs_for_common(self):
        config, input_ids = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch_accelerator
@require_torch
class FastSpeech2ConformerWithHifiGanTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FastSpeech2ConformerWithHifiGan,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_resize_embeddings = False
    is_encoder_decoder = True

    def setUp(self):
        self.model_tester = FastSpeech2ConformerWithHifiGanTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    msg = f"Parameter {name} of model {model_class} seems not properly initialized"
                    if "norm" in name:
                        if "bias" in name:
                            self.assertEqual(param.data.mean().item(), 0.0, msg=msg)
                        if "weight" in name:
                            self.assertEqual(param.data.mean().item(), 1.0, msg=msg)
                    elif "conv" in name or "embed" in name:
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0, msg=msg)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        return inputs_dict

    def test_duration_energy_pitch_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.model_config.return_dict = True

        seq_len = self.model_tester.seq_length
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # duration
            self.assertListEqual(list(outputs.duration_outputs.shape), [self.model_tester.batch_size, seq_len])
            # energy
            self.assertListEqual(list(outputs.energy_outputs.shape), [self.model_tester.batch_size, seq_len, 1])
            # pitch
            self.assertListEqual(list(outputs.pitch_outputs.shape), [self.model_tester.batch_size, seq_len, 1])

    def test_hidden_states_output(self):
        def _check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            for idx, hidden_states in enumerate([outputs.encoder_hidden_states, outputs.decoder_hidden_states]):
                expected_num_layers = getattr(
                    self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
                )

                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertIsInstance(hidden_states, (list, tuple))
                expected_batch_size, expected_seq_length, expected_hidden_size = hidden_states[0].shape
                self.assertEqual(expected_batch_size, self.model_tester.batch_size)
                # Only test encoder seq_length since decoder seq_length is variable based on inputs
                if idx == 0:
                    self.assertEqual(expected_seq_length, self.model_tester.seq_length)
                self.assertEqual(expected_hidden_size, self.model_tester.hidden_size)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict["output_hidden_states"] = True
        _check_hidden_states_output(inputs_dict, config, FastSpeech2ConformerWithHifiGan)

        # check that output_hidden_states also work using config
        del inputs_dict["output_hidden_states"]
        config.model_config.output_hidden_states = True

        _check_hidden_states_output(inputs_dict, config, FastSpeech2ConformerWithHifiGan)

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = FastSpeech2ConformerWithHifiGan(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            _, info = FastSpeech2ConformerWithHifiGan.from_pretrained(tmpdirname, output_loading_info=True)
        self.assertEqual(info["missing_keys"], [])

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = FastSpeech2ConformerWithHifiGan(config)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = [
            "input_ids",
            "attention_mask",
            "spectrogram_labels",
            "duration_labels",
            "pitch_labels",
            "energy_labels",
            "speaker_ids",
            "lang_ids",
            "speaker_embedding",
            "return_dict",
            "output_attentions",
            "output_hidden_states",
        ]
        self.assertListEqual(arg_names, expected_arg_names)

    # Override as FastSpeech2Conformer does not output cross attentions
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.model_config.output_hidden_states = True
        config.model_config.output_attentions = True

        model = FastSpeech2ConformerWithHifiGan(config)
        model.to(torch_device)
        model.eval()

        inputs = self._prepare_for_class(inputs_dict, FastSpeech2ConformerModel)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        decoder_hidden_states = outputs.decoder_hidden_states[0]
        decoder_hidden_states.retain_grad()

        encoder_attentions = outputs.encoder_attentions[0]
        encoder_attentions.retain_grad()

        decoder_attentions = outputs.decoder_attentions[0]
        decoder_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(decoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(decoder_attentions.grad)

    def test_attention_outputs(self):
        """
        Custom `test_attention_outputs` since FastSpeech2Conformer does not output cross attentions, has variable
        decoder attention shape, and uniquely outputs energy, pitch, and durations.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.model_config.return_dict = True

        seq_len = self.model_tester.seq_length

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.model_config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.encoder_attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.model_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            encoder_attentions = outputs.encoder_attentions
            self.assertEqual(len(encoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(encoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            correct_outlen = 8
            self.assertEqual(out_len, correct_outlen)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )

    @slow
    def test_model_from_pretrained(self):
        model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        self.assertIsNotNone(model)

    @unittest.skip(reason="FastSpeech2Conformer does not accept inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="FastSpeech2Conformer has no input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        "FastSpeech2Conformer predicts durations in linear domain during inference"
        "Even small differences on hidden states lead to different durations, due to `torch.round`"
    )
    def test_batching_equivalence(self):
        pass


@require_torch
@require_g2p_en
@slow
class FastSpeech2ConformerWithHifiGanIntegrationTest(unittest.TestCase):
    def test_inference_integration(self):
        model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")
        model.to(torch_device)
        model.eval()

        tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        text = "Test that this generates speech"
        input_ids = tokenizer(text, return_tensors="pt").to(torch_device)["input_ids"]

        output = model(input_ids)
        waveform = output.waveform

        # waveform is too large (1, 52480), so only check first 100 elements
        # fmt: off
        expected_waveform = torch.tensor(
            [
                [-9.6345e-04,  1.3557e-03,  5.7559e-04,  2.4706e-04,  2.2675e-04, 1.2258e-04,  4.7784e-04,  1.0109e-03, -1.9718e-04,  6.3495e-04, 3.2106e-04,  6.3620e-05,  9.1713e-04, -2.5664e-05,  1.9596e-04, 6.0418e-04,  8.1112e-04,  3.6342e-04, -6.3396e-04, -2.0146e-04, -1.1768e-04,  4.3155e-04,  7.5599e-04, -2.2972e-04, -9.5665e-05, 3.3078e-04,  1.3793e-04, -1.4932e-04, -3.9645e-04,  3.6473e-05, -1.7224e-04, -4.5370e-05, -4.8950e-04, -4.3059e-04,  1.0451e-04, -1.0485e-03, -6.0410e-04,  1.6990e-04, -2.1997e-04, -3.8769e-04, -7.6898e-04, -3.2372e-04, -1.9783e-04,  5.2896e-05, -1.0586e-03, -7.8516e-04,  7.6867e-04, -8.5331e-05, -4.8158e-04, -4.5362e-05, -1.0770e-04,  6.6823e-04,  3.0765e-04,  3.3669e-04,  9.5677e-04, 1.0458e-03,  5.8129e-04,  3.3737e-04,  1.0816e-03,  7.0346e-04, 4.2378e-04,  4.3131e-04,  2.8095e-04,  1.2201e-03,  5.6121e-04, -1.1086e-04,  4.9908e-04,  1.5586e-04,  4.2046e-04, -2.8088e-04, -2.2462e-04, -1.5539e-04, -7.0126e-04, -2.8577e-04, -3.3693e-04, -1.2471e-04, -6.9104e-04, -1.2867e-03, -6.2651e-04, -2.5586e-04, -1.3201e-04, -9.4537e-04, -4.8438e-04,  4.1458e-04,  6.4109e-04, 1.0891e-04, -6.3764e-04,  4.5573e-04,  8.2974e-04,  3.2973e-06, -3.8274e-04, -2.0400e-04,  4.9922e-04,  2.1508e-04, -1.1009e-04, -3.9763e-05,  3.0576e-04,  3.1485e-05, -2.7574e-05,  3.3856e-04],
            ],
            device=torch_device,
        )
        # fmt: on

        torch.testing.assert_close(waveform[0, :100], expected_waveform, rtol=1e-4, atol=1e-4)
        self.assertEqual(waveform.shape, (1, 52480))
