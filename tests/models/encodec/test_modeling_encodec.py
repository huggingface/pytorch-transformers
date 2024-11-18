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
"""Testing suite for the PyTorch Encodec model."""

import copy
import inspect
import os
import random
import tempfile
import unittest

import numpy as np
import torch.nn.functional as F
import torchaudio
from datasets import Audio, load_dataset

from transformers import AutoProcessor, EncodecConfig
from transformers.models.encodec.loss_encodec import (
    Balancer,
    compute_discriminator_loss,
    compute_feature_matching_loss,
    compute_generator_adv_loss,
)
from transformers.testing_utils import is_torch_available, require_torch, require_torchaudio, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import EncodecDiscriminator, EncodecDiscriminatorConfig, EncodecModel


def prepare_inputs_dict(
    config,
    input_ids=None,
    input_values=None,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if input_ids is not None:
        encoder_dict = {"input_ids": input_ids}
    else:
        encoder_dict = {"input_values": input_values}

    decoder_dict = {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}

    return {**encoder_dict, **decoder_dict}


@require_torch
class EncodecModelTester:
    def __init__(
        self,
        parent,
        # `batch_size` needs to be an even number if the model has some outputs with batch dim != 0.
        batch_size=12,
        num_channels=2,
        is_training=False,
        intermediate_size=40,
        hidden_size=32,
        num_filters=8,
        num_residual_layers=1,
        upsampling_ratios=[8, 4],
        num_lstm_layers=1,
        codebook_size=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios
        self.num_lstm_layers = num_lstm_layers
        self.codebook_size = codebook_size

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        if model_class == EncodecDiscriminator:
            config = EncodecDiscriminatorConfig()
            inputs_dict = {
                "input_values": floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
            }
        else:
            config = self.get_config()
            inputs_dict = self.prepare_config_and_inputs()[1]

        if model_class == EncodecDiscriminator:
            inputs_dict["input_values"] = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        else:
            inputs_dict["audio_codes"] = ids_tensor([1, self.batch_size, 1, self.num_channels], self.codebook_size).type(torch.int32)
            inputs_dict["audio_scales"] = [None]

        return config, inputs_dict

    def get_config(self):
        return EncodecConfig(
            audio_channels=self.num_channels,
            chunk_in_sec=None,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            num_residual_layers=self.num_residual_layers,
            upsampling_ratios=self.upsampling_ratios,
            num_lstm_layers=self.num_lstm_layers,
            codebook_size=self.codebook_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = EncodecModel(config=config).to(torch_device).eval()

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size, self.num_channels, self.intermediate_size)
        )


@require_torch
class EncodecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EncodecModel, EncodecDiscriminator) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    pipeline_model_mapping = {"feature-extraction": EncodecModel} if is_torch_available() else {}

    # Test copied from: https://github.com/facebookresearch/encodec/blob/main/encodec/msstftd.py#L132
    @slow
    def test_discriminator_output_shapes_and_feature_maps(self):
        disc = EncodecDiscriminator(EncodecDiscriminatorConfig())
        y = torch.randn(1, 1, 24000)
        y_hat = torch.randn(1, 1, 24000)

        y_disc_r, fmap_r = disc(y)
        y_disc_gen, fmap_gen = disc(y_hat)
        assert len(y_disc_r) == len(y_disc_gen) == len(fmap_r) == len(fmap_gen) == disc.num_discriminators

        assert all(len(fm) == 5 for fm in fmap_r + fmap_gen)
        assert all(list(f.shape)[:2] == [1, 32] for fm in fmap_r + fmap_gen for f in fm)
        assert all(len(logits.shape) == 4 for logits in y_disc_r + y_disc_gen)

    # Test copied from: https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py#L121
    @slow
    def test_balancer_basic(self):
        x = torch.zeros(1, requires_grad=True)
        one = torch.ones_like(x)
        loss_1 = F.l1_loss(x, one)
        loss_2 = 100 * F.l1_loss(x, -one)
        losses = {"1": loss_1, "2": loss_2}

        balancer = Balancer(weights={"1": 1, "2": 1}, rescale_grads=False)
        balancer.backward(losses, x)
        assert torch.allclose(x.grad, torch.tensor(99.0)), x.grad

        loss_1 = F.l1_loss(x, one)
        loss_2 = 100 * F.l1_loss(x, -one)
        losses = {"1": loss_1, "2": loss_2}
        x.grad = None
        balancer = Balancer(weights={"1": 1, "2": 1}, rescale_grads=True)
        balancer.backward({"1": loss_1, "2": loss_2}, x)
        assert torch.allclose(x.grad, torch.tensor(0.0)), x.grad

    @slow
    @require_torchaudio
    def test_training_with_discriminator(self):
        model_id = "facebook/encodec_24khz"
        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        model.train()

        discriminator_config = EncodecDiscriminatorConfig()
        discriminator = EncodecDiscriminator(discriminator_config).to(torch_device)
        discriminator.train()

        generator_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Generate a sine wave input
        sample_rate = 24000
        duration = 1
        t = torch.linspace(0, duration, int(sample_rate * duration), device=torch_device)
        frequency = 440  # A4 note
        audio_input = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0).unsqueeze(0)
        audio_input = audio_input.repeat(1, model.config.audio_channels, 1).to(torch_device)

        inputs = processor(
            raw_audio=audio_input.squeeze().cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(torch_device)

        num_epochs = 3

        loss_weights = {"reconstruction_loss": 1.0, "g_adv_loss": 3.0, "fm_loss": 3.0}

        balancer = Balancer(
            weights=loss_weights,
            rescale_grads=True,
            total_norm=1.0,
            ema_decay=0.999,
            per_batch_item=True,
            epsilon=1e-12,
            monitor=False,
        )

        for epoch in range(num_epochs):
            real_audio = input_values

            # Update discriminator based on the probability outlined in the paper
            if sample_rate == 24000:
                update_discriminator = random.random() < (2 / 3)
            elif sample_rate == 48000:
                update_discriminator = random.random() < 0.5
            else:
                raise ValueError("Unsupported sample rate")

            if update_discriminator:
                discriminator_optimizer.zero_grad()

                # Generate fake audio with the generator
                with torch.no_grad():
                    outputs = model(input_values, return_dict=True)
                fake_audio = outputs.audio_values.detach()  # Detach to prevent gradients flowing to the generator

                real_logits, _ = discriminator(real_audio)
                fake_logits, _ = discriminator(fake_audio)

                discriminator_loss = compute_discriminator_loss(
                    real_logits=real_logits,
                    fake_logits=fake_logits,
                    num_discriminators=discriminator.num_discriminators,
                )

                discriminator_loss.backward()
                discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()

            # Generate fake audio and compute losses
            outputs = model(input_values, return_dict=True)
            fake_audio = outputs.audio_values

            # Compute generator adversarial loss and feature matching loss
            fake_logits, fake_features = discriminator(fake_audio)
            _, real_features = discriminator(real_audio)

            # Generator adversarial loss
            g_adv_loss = compute_generator_adv_loss(
                fake_logits=fake_logits, num_discriminators=discriminator.num_discriminators
            )

            # Feature matching loss
            fm_loss = compute_feature_matching_loss(
                real_features=real_features,
                fake_features=fake_features,
                num_discriminators=discriminator.num_discriminators,
            )

            losses_to_balance = {
                "reconstruction_loss": outputs.reconstruction_loss,
                "g_adv_loss": g_adv_loss,
                "fm_loss": fm_loss,
            }

            # Model output (the reconstructed audio)
            model_output = outputs.audio_values

            balancer.backward(losses=losses_to_balance, input=model_output)

            if outputs.commitment_loss is not None:
                outputs.commitment_loss.backward()

            generator_optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}")
            if update_discriminator:
                print(f"Discriminator loss: {discriminator_loss.item():.4f}")
            else:
                print("Discriminator not updated this epoch")
            print(f"Generator adversarial loss: {g_adv_loss.item():.4f}")
            print(f"Feature matching loss: {fm_loss.item():.4f}")
            print(f"Reconstruction loss: {outputs.reconstruction_loss.item():.4f}")
            if outputs.commitment_loss is not None:
                print(f"Commitment loss: {outputs.commitment_loss.item():.4f}")
            total_gen_loss = outputs.reconstruction_loss.item() + g_adv_loss.item() + fm_loss.item()
            if outputs.commitment_loss is not None:
                total_gen_loss += outputs.commitment_loss.item()
            print(f"Total generator loss (before balancing): {total_gen_loss:.4f}\n")

    @slow
    @require_torchaudio
    def test_reconstruction_loss(self):
        model_id = "facebook/encodec_24khz"
        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        model.eval()

        sample_rate = 24000
        duration = 1
        t = torch.linspace(0, duration, sample_rate)
        frequency = 440  # A4 note
        audio_input = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0).unsqueeze(0)
        audio_input = audio_input.repeat(1, model.config.audio_channels, 1).to(torch_device)

        inputs = processor(
            raw_audio=audio_input.squeeze().cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(torch_device)

        bandwidths = [1.5, 6.0, 12.0, 24.0]
        for bandwidth in bandwidths:
            with torch.no_grad():
                outputs = model(input_values, bandwidth=bandwidth, return_dict=True, return_loss=True)

            print(f"\nBandwidth: {bandwidth}")
            print(f"Reconstruction loss: {outputs.reconstruction_loss.item()}")
            print(f"Audio codes shape: {outputs.audio_codes[0].shape}")
            print(f"Audio values shape: {outputs.audio_values.shape}")
            print(f"Input max: {input_values.max().item()}, min: {input_values.min().item()}")
            print(f"Output max: {outputs.audio_values.max().item()}, min: {outputs.audio_values.min().item()}")

            reconstructed_audio = outputs.audio_values
            mae = torch.mean(torch.abs(input_values - reconstructed_audio))
            print(f"Mean Absolute Error (MAE): {mae.item()}")

            # Compare spectrograms
            spec_transform = torchaudio.transforms.Spectrogram().to(torch_device)
            input_spec = spec_transform(input_values.squeeze())
            output_spec = spec_transform(reconstructed_audio.squeeze())
            spec_mae = torch.mean(torch.abs(input_spec - output_spec))
            print(f"Spectrogram MAE: {spec_mae.item()}")

    @slow
    @require_torchaudio
    def test_gradients_exist(self):
        model_id = "facebook/encodec_24khz"
        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        model.train()

        sample_rate = 24000
        duration = 1
        t = torch.linspace(0, duration, int(sample_rate * duration), device=torch_device)
        frequency = 440  # A4 note
        audio_input = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0).unsqueeze(0)
        audio_input = audio_input.repeat(1, model.config.audio_channels, 1).to(torch_device)

        inputs = processor(
            raw_audio=audio_input.squeeze().cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(torch_device)

        outputs = model(input_values, return_dict=True, return_loss=True)
        total_loss = outputs.reconstruction_loss

        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN values")

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not have attention and does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = EncodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=EncodecConfig, hidden_size=37, common_properties=[], has_text_modality=False
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

            expected_arg_names = ["input_values", "padding_mask", "bandwidth"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="The EncodecModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The EncodecModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `attention` logic"
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `attention` logic"
    )
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_torchscript_output_hidden_state(self):
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

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

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `attention` logic"
    )
    def test_attention_outputs(self):
        pass

    def test_feed_forward_chunking(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            config = copy.deepcopy(original_config)
            config.chunk_length_s = None
            config.overlap = None
            config.sampling_rate = 10

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs["input_values"] = inputs["input_values"].repeat(1, 1, 10)

            hidden_states_no_chunk = model(**inputs)[0]

            torch.manual_seed(0)
            config.chunk_length_s = 1
            config.overlap = 0
            config.sampling_rate = 10

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**inputs)[0]
            self.assertTrue(torch.allclose(hidden_states_no_chunk, hidden_states_with_chunk, atol=1e-3))

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
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
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs)

                self.assertTrue(isinstance(tuple_output, tuple))
                self.assertTrue(isinstance(dict_output, dict))

                for tuple_value, dict_value in zip(tuple_output, dict_output.values()):
                    self.assertTrue(
                        torch.allclose(
                            set_nan_tensor_to_zero(tuple_value), set_nan_tensor_to_zero(dict_value), atol=1e-5
                        ),
                        msg=(
                            "Tuple and dict output are not equal. Difference:"
                            f" {torch.max(torch.abs(tuple_value - dict_value))}. Tuple has `nan`:"
                            f" {torch.isnan(tuple_value).any()} and `inf`: {torch.isinf(tuple_value)}. Dict has"
                            f" `nan`: {torch.isnan(dict_value).any()} and `inf`: {torch.isinf(dict_value)}."
                        ),
                    )

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            configs_no_init = _config_zero_init(config)
            model = model_class(config=configs_no_init)

            for name, param in model.named_parameters():
                uniform_init_parms = ["conv"]
                ignore_init = ["lstm"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif not any(x in name for x in ignore_init):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
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
class EncodecIntegrationTest(unittest.TestCase):
    def test_integration_24kHz(self):
        expected_rmse = {
            "1.5": 0.0025,
            "24.0": 0.0015,
        }
        expected_codesums = {
            "1.5": [371955],
            "24.0": [6659962],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "facebook/encodec_24khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        for bandwidth, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                encoder_outputs = model.encode(inputs["input_values"], bandwidth=float(bandwidth))

                audio_code_sums = [a[0].sum().cpu().item() for a in encoder_outputs[0]]

                # make sure audio encoded codes are correct
                self.assertListEqual(audio_code_sums, expected_codesums[bandwidth])

                audio_codes, scales = encoder_outputs.to_tuple()
                input_values_dec = model.decode(audio_codes, scales, inputs["padding_mask"])[0]
                input_values_enc_dec = model(
                    inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth)
                )[-1]

            # make sure forward and decode gives same result
            self.assertTrue(torch.allclose(input_values_dec, input_values_enc_dec, atol=1e-3))

            # make sure shape matches
            self.assertTrue(inputs["input_values"].shape == input_values_enc_dec.shape)

            arr = inputs["input_values"][0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_48kHz(self):
        expected_rmse = {
            "3.0": 0.001,
            "24.0": 0.0005,
        }
        expected_codesums = {
            "3.0": [144259, 146765, 156435, 176871, 161971],
            "24.0": [1568553, 1294948, 1306190, 1464747, 1663150],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "facebook/encodec_48khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        model = model.eval()
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        # transform mono to stereo
        audio_sample = np.array([audio_sample, audio_sample])

        inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt").to(
            torch_device
        )

        for bandwidth, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                encoder_outputs = model.encode(
                    inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth), return_dict=False
                )
                audio_code_sums = [a[0].sum().cpu().item() for a in encoder_outputs[0]]

                # make sure audio encoded codes are correct
                self.assertListEqual(audio_code_sums, expected_codesums[bandwidth])
                audio_codes, scales = encoder_outputs
                input_values_dec = model.decode(audio_codes, scales, inputs["padding_mask"])[0]
                input_values_enc_dec = model(
                    inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth)
                )[-1]

            # make sure forward and decode gives same result
            self.assertTrue(torch.allclose(input_values_dec, input_values_enc_dec, atol=1e-3))

            # make sure shape matches
            self.assertTrue(inputs["input_values"].shape == input_values_enc_dec.shape)

            arr = inputs["input_values"][0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)

    def test_batch_48kHz(self):
        expected_rmse = {
            "3.0": 0.001,
            "24.0": 0.0005,
        }
        expected_codesums = {
            "3.0": [
                [72410, 79137, 76694, 90854, 73023, 82980, 72707, 54842],
                [85561, 81870, 76953, 48967, 79315, 85442, 81479, 107241],
            ],
            "24.0": [
                [72410, 79137, 76694, 90854, 73023, 82980, 72707, 54842],
                [85561, 81870, 76953, 48967, 79315, 85442, 81479, 107241],
            ],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "facebook/encodec_48khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id, chunk_length_s=1, overlap=0.01)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

        audio_samples = [
            np.array([audio_sample["array"], audio_sample["array"]])
            for audio_sample in librispeech_dummy[-2:]["audio"]
        ]

        inputs = processor(raw_audio=audio_samples, sampling_rate=processor.sampling_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(torch_device)
        for bandwidth, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                encoder_outputs = model.encode(input_values, bandwidth=float(bandwidth), return_dict=False)
                audio_code_sums_0 = [a[0][0].sum().cpu().item() for a in encoder_outputs[0]]
                audio_code_sums_1 = [a[0][1].sum().cpu().item() for a in encoder_outputs[0]]

                # make sure audio encoded codes are correct
                self.assertListEqual(audio_code_sums_0, expected_codesums[bandwidth][0])
                self.assertListEqual(audio_code_sums_1, expected_codesums[bandwidth][1])

                audio_codes, scales = encoder_outputs
                input_values_dec = model.decode(audio_codes, scales)[0]
                input_values_enc_dec = model(input_values, bandwidth=float(bandwidth))[-1]

            # make sure forward and decode gives same result
            self.assertTrue(torch.allclose(input_values_dec, input_values_enc_dec, atol=1e-3))

            # make sure shape matches
            self.assertTrue(input_values.shape == input_values_enc_dec.shape)

            arr = input_values[0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)
