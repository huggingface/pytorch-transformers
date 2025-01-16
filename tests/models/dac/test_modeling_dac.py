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
"""Testing suite for the PyTorch Dac model."""

import inspect
import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np
from datasets import Audio, load_dataset

from transformers import AutoProcessor, DacConfig, DacModel
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


@require_torch
# Copied from transformers.tests.encodec.test_modeling_encodec.EncodecModelTester with Encodec->Dac
class DacModelTester:
    # Ignore copy
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=1,
        is_training=False,
        intermediate_size=1024,
        encoder_hidden_size=16,
        downsampling_ratios=[2, 4, 4],
        decoder_hidden_size=16,
        n_codebooks=6,
        codebook_size=512,
        codebook_dim=4,
        quantizer_dropout=0.0,
        commitment_loss_weight=0.25,
        codebook_loss_weight=1.0,
        sample_rate=16000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.sample_rate = sample_rate

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_hidden_size = decoder_hidden_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.commitment_loss_weight = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}

        return config, inputs_dict

    # Ignore copy
    def get_config(self):
        return DacConfig(
            encoder_hidden_size=self.encoder_hidden_size,
            downsampling_ratios=self.downsampling_ratios,
            decoder_hidden_size=self.decoder_hidden_size,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_dropout=self.quantizer_dropout,
            commitment_loss_weight=self.commitment_loss_weight,
            codebook_loss_weight=self.codebook_loss_weight,
        )

    # Ignore copy
    def create_and_check_model_forward(self, config, inputs_dict):
        model = DacModel(config=config).to(torch_device).eval()

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(result.audio_values.shape, (self.batch_size, self.intermediate_size))


@require_torch
# Copied from transformers.tests.encodec.test_modeling_encodec.EncodecModelTest with Encodec->Dac
class DacModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DacModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    pipeline_model_mapping = {"feature-extraction": DacModel} if is_torch_available() else {}

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not have attention and does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = DacModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=DacConfig, hidden_size=37, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            # Ignore copy
            expected_arg_names = ["input_values", "n_quantizers", "return_dict"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip("The DacModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(main_input)
                traced_model = torch.jit.trace(model, main_input)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip("The DacModel is not transformers based, thus it does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip("No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip("No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # outputs are not tensors but list (since each sequence don't have the same frame_length)
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    # Ignore copy
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv", "in_proj", "out_proj", "codebook"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)


def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


def compute_rmse(arr1, arr2):
    arr1_normalized = normalize(arr1)
    arr2_normalized = normalize(arr2)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_torch
class DacIntegrationTest(unittest.TestCase):
    def test_integration_16khz(self):
        expected_rmse = 0.004

        expected_encoder_sums_dict = {
            "loss": 24.8596,
            "quantized_representation": -0.0745,
            "audio_codes": 504.0948,
            "projected_latents": 0.0682,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_name = "dac_16khz"

        model_id = "descript/{}".format(model_name)
        model = DacModel.from_pretrained(model_id, force_download=True).to(torch_device).eval()
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[0]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])

            expected_encoder_sums = torch.tensor(list(expected_encoder_sums_dict.values()), dtype=torch.float32)
            encoder_outputs_mean = torch.tensor([v.float().mean().cpu().item() for v in encoder_outputs.to_tuple()])

            # make sure audio encoded codes are correct
            torch.testing.assert_close(encoder_outputs_mean, expected_encoder_sums, rtol=1e-3, atol=1e-3)

            _, quantized_representation, _, _ = encoder_outputs.to_tuple()
            input_values_dec = model.decode(quantized_representation)[0]
            input_values_enc_dec = model(inputs["input_values"])[1]

            # make sure forward and decode gives same result
            torch.testing.assert_close(input_values_dec, input_values_enc_dec, rtol=1e-3, atol=1e-3)

            arr = inputs["input_values"][0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            max_length = min(arr_enc_dec.shape[-1], arr.shape[-1])

            arr_cut = arr[0, :max_length].copy()
            arr_enc_dec_cut = arr_enc_dec[:max_length].copy()

            # make sure audios are more or less equal
            rmse = compute_rmse(arr_cut, arr_enc_dec_cut)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_24khz(self):
        expected_rmse = 0.0039

        expected_encoder_output_dict = {
            "quantized_representation": torch.tensor([0.6257, 3.1245, 5.2514, 2.3160, 1.5774]),
            "audio_codes": torch.tensor([919, 919, 234, 777, 234]),
            "projected_latents": torch.tensor([-4.7841, -5.0063, -4.5595, -5.0372, -5.4280]),
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_name = "dac_24khz"

        model_id = "descript/{}".format(model_name)
        model = DacModel.from_pretrained(model_id, force_download=True).to(torch_device).eval()
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[0]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])

            expected_quantized_representation = encoder_outputs["quantized_representation"][0, 0, :5].cpu()
            expected_audio_codes = encoder_outputs["audio_codes"][0, 0, :5].cpu()
            expected_projected_latents = encoder_outputs["projected_latents"][0, 0, :5].cpu()

            # make sure values are correct for audios slices
            self.assertTrue(
                torch.allclose(
                    expected_quantized_representation,
                    expected_encoder_output_dict["quantized_representation"],
                    atol=1e-3,
                )
            )
            self.assertTrue(
                torch.allclose(expected_audio_codes, expected_encoder_output_dict["audio_codes"], atol=1e-3)
            )
            self.assertTrue(
                torch.allclose(
                    expected_projected_latents, expected_encoder_output_dict["projected_latents"], atol=1e-3
                )
            )

            _, quantized_representation, _, _ = encoder_outputs.to_tuple()
            input_values_dec = model.decode(quantized_representation)[0]
            input_values_enc_dec = model(inputs["input_values"])[1]

            input_values_from_codes = model.decode(audio_codes=encoder_outputs.audio_codes)[0]

            # make sure decode from audio codes and quantized values give more or less the same results
            torch.testing.assert_close(input_values_from_codes, input_values_dec, rtol=1e-5, atol=1e-5)

            # make sure forward and decode gives same result
            torch.testing.assert_close(input_values_dec, input_values_enc_dec, rtol=1e-3, atol=1e-3)

            arr = inputs["input_values"][0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            max_length = min(arr_enc_dec.shape[-1], arr.shape[-1])

            arr_cut = arr[0, :max_length].copy()
            arr_enc_dec_cut = arr_enc_dec[:max_length].copy()

            # make sure audios are more or less equal
            rmse = compute_rmse(arr_cut, arr_enc_dec_cut)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_44khz(self):
        expected_rmse = 0.002

        expected_encoder_sums_dict = {
            "loss": 34.3612,
            "quantized_representation": 0.0078,
            "audio_codes": 509.6812,
            "projected_latents": -0.1054,
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_name = "dac_44khz"

        model_id = "descript/{}".format(model_name)
        model = DacModel.from_pretrained(model_id).to(torch_device).eval()
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[0]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])

            expected_encoder_sums = torch.tensor(list(expected_encoder_sums_dict.values()), dtype=torch.float32)
            encoder_outputs_mean = torch.tensor([v.float().mean().cpu().item() for v in encoder_outputs.to_tuple()])

            # make sure audio encoded codes are correct
            torch.testing.assert_close(encoder_outputs_mean, expected_encoder_sums, rtol=1e-3, atol=1e-3)

            _, quantized_representation, _, _ = encoder_outputs.to_tuple()
            input_values_dec = model.decode(quantized_representation)[0]
            input_values_enc_dec = model(inputs["input_values"])[1]

            # make sure forward and decode gives same result
            torch.testing.assert_close(input_values_dec, input_values_enc_dec, rtol=1e-3, atol=1e-3)

            arr = inputs["input_values"][0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            max_length = min(arr_enc_dec.shape[-1], arr.shape[-1])

            arr_cut = arr[0, :max_length].copy()
            arr_enc_dec_cut = arr_enc_dec[:max_length].copy()

            # make sure audios are more or less equal
            rmse = compute_rmse(arr_cut, arr_enc_dec_cut)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_batch_16khz(self):
        expected_rmse = 0.002

        expected_encoder_sums_dict = {
            "loss": 20.3913,
            "quantized_representation": -0.0538,
            "audio_codes": 487.8470,
            "projected_latents": 0.0237,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_name = "dac_16khz"

        model_id = "descript/{}".format(model_name)
        model = DacModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

        audio_samples = [np.array([audio_sample["array"]])[0] for audio_sample in librispeech_dummy[-2:]["audio"]]

        inputs = processor(
            raw_audio=audio_samples,
            sampling_rate=processor.sampling_rate,
            truncation=False,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])

            expected_encoder_sums = torch.tensor(list(expected_encoder_sums_dict.values()), dtype=torch.float32)
            encoder_outputs_mean = torch.tensor([v.float().mean().item() for v in encoder_outputs.to_tuple()])

            # make sure audio encoded codes are correct
            torch.testing.assert_close(encoder_outputs_mean, expected_encoder_sums, rtol=1e-3, atol=1e-3)

            _, quantized_representation, _, _ = encoder_outputs.to_tuple()
            input_values_dec = model.decode(quantized_representation)[0]
            input_values_enc_dec = model(inputs["input_values"])[1]

            # make sure forward and decode gives same result
            torch.testing.assert_close(input_values_dec, input_values_enc_dec, rtol=1e-3, atol=1e-3)

            arr = inputs["input_values"].cpu().numpy()
            arr_enc_dec = input_values_enc_dec.cpu().numpy()

            max_length = min(arr_enc_dec.shape[-1], arr.shape[-1])

            arr_cut = arr[:, 0, :max_length].copy()
            arr_enc_dec_cut = arr_enc_dec[:, :max_length].copy()

            # make sure audios are more or less equal
            rmse = compute_rmse(arr_cut, arr_enc_dec_cut)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_batch_24khz(self):
        expected_rmse = 0.002

        expected_encoder_sums_dict = {
            "loss": 24.2309,
            "quantized_representation": 0.0520,
            "audio_codes": 510.2700,
            "projected_latents": -0.0076,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_name = "dac_24khz"

        model_id = "descript/{}".format(model_name)
        model = DacModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

        audio_samples = [np.array([audio_sample["array"]])[0] for audio_sample in librispeech_dummy[-2:]["audio"]]

        inputs = processor(
            raw_audio=audio_samples,
            sampling_rate=processor.sampling_rate,
            truncation=False,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])

            expected_encoder_sums = torch.tensor(list(expected_encoder_sums_dict.values()), dtype=torch.float32)
            encoder_outputs_mean = torch.tensor([v.float().mean().cpu().item() for v in encoder_outputs.to_tuple()])

            # make sure audio encoded codes are correct
            torch.testing.assert_close(encoder_outputs_mean, expected_encoder_sums, rtol=1e-3, atol=1e-3)

            _, quantized_representation, _, _ = encoder_outputs.to_tuple()
            input_values_dec = model.decode(quantized_representation)[0]
            input_values_enc_dec = model(inputs["input_values"])[1]

            # make sure forward and decode gives same result
            torch.testing.assert_close(input_values_dec, input_values_enc_dec, rtol=1e-3, atol=1e-3)

            arr = inputs["input_values"].cpu().numpy()
            arr_enc_dec = input_values_enc_dec.cpu().numpy()

            max_length = min(arr_enc_dec.shape[-1], arr.shape[-1])

            arr_cut = arr[:, 0, :max_length].copy()
            arr_enc_dec_cut = arr_enc_dec[:, :max_length].copy()

            # make sure audios are more or less equal
            rmse = compute_rmse(arr_cut, arr_enc_dec_cut)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_batch_44khz(self):
        expected_rmse = 0.001

        expected_encoder_sums_dict = {
            "loss": 25.9233,
            "quantized_representation": 0.0013,
            "audio_codes": 528.5620,
            "projected_latents": -0.1194,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_name = "dac_44khz"

        model_id = "descript/{}".format(model_name)
        model = DacModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

        audio_samples = [np.array([audio_sample["array"]])[0] for audio_sample in librispeech_dummy[-2:]["audio"]]

        inputs = processor(
            raw_audio=audio_samples,
            sampling_rate=processor.sampling_rate,
            truncation=False,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])

            expected_encoder_sums = torch.tensor(list(expected_encoder_sums_dict.values()), dtype=torch.float32)
            encoder_outputs_mean = torch.tensor([v.float().mean().cpu().item() for v in encoder_outputs.to_tuple()])

            # make sure audio encoded codes are correct
            torch.testing.assert_close(encoder_outputs_mean, expected_encoder_sums, rtol=1e-3, atol=1e-3)

            _, quantized_representation, _, _ = encoder_outputs.to_tuple()
            input_values_dec = model.decode(quantized_representation)[0]
            input_values_enc_dec = model(inputs["input_values"])[1]

            # make sure forward and decode gives same result
            torch.testing.assert_close(input_values_dec, input_values_enc_dec, rtol=1e-3, atol=1e-3)

            arr = inputs["input_values"].cpu().numpy()
            arr_enc_dec = input_values_enc_dec.cpu().numpy()

            max_length = min(arr_enc_dec.shape[-1], arr.shape[-1])

            arr_cut = arr[:, 0, :max_length].copy()
            arr_enc_dec_cut = arr_enc_dec[:, :max_length].copy()

            # make sure audios are more or less equal
            rmse = compute_rmse(arr_cut, arr_enc_dec_cut)
            self.assertTrue(rmse < expected_rmse)
