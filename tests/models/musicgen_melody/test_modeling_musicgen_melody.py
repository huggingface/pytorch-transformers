# coding=utf-8
# Copyright 2024, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Musicgen Melody model."""

import copy
import inspect
import math
import tempfile
import unittest

import numpy as np
from parameterized import parameterized
from pytest import mark

from transformers import (
    EncodecConfig,
    MusicgenMelodyConfig,
    MusicgenMelodyDecoderConfig,
    PretrainedConfig,
    T5Config,
)
from transformers.testing_utils import (
    is_torch_available,
    is_torchaudio_available,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_torch_gpu,
    require_torch_sdpa,
    require_torchaudio,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, sdpa_kernel
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MusicgenMelodyForCausalLM,
        MusicgenMelodyForConditionalGeneration,
        MusicgenMelodyModel,
        set_seed,
    )

if is_torchaudio_available():
    from transformers import MusicgenMelodyProcessor


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def prepare_musicgen_melody_decoder_inputs_dict(
    config,
    input_ids,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.reshape(-1, config.num_codebooks, input_ids.shape[-1])[:, 0, :]
        attention_mask = attention_mask.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    if encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=torch_device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "head_mask": head_mask,
    }


class MusicgenMelodyDecoderTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers because of #29297
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
        conditional_seq_length=4,
        audio_channels=1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.conditional_seq_length = conditional_seq_length
        self.encoder_seq_length = conditional_seq_length + seq_length
        self.audio_channels = audio_channels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)
        encoder_hidden_states = floats_tensor([self.batch_size, self.conditional_seq_length, self.hidden_size])

        config = self.get_config()
        inputs_dict = prepare_musicgen_melody_decoder_inputs_dict(
            config,
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        return config, inputs_dict

    def get_config(self):
        config = MusicgenMelodyDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
            audio_channels=self.audio_channels,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MusicgenMelodyDecoderTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenMelodyModel, MusicgenMelodyForCausalLM) if is_torch_available() else ()
    greedy_sample_model_classes = (
        (MusicgenMelodyForCausalLM,) if is_torch_available() else ()
    )  # the model uses a custom generation method so we only run a specific subset of the generation tests
    test_pruning = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = MusicgenMelodyDecoderTester(self)
        self.config_tester = ConfigTester(self, config_class=MusicgenMelodyDecoderConfig, hidden_size=16)

    def test_config(self):
        self.config_tester.run_common_tests()

    # special case for labels
    # Copied from tests.models.musicgen.test_modeling_musicgen.MusicgenDecoderTest._prepare_for_class
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_codebooks),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    # Copied from tests.models.musicgen.test_modeling_musicgen.MusicgenDecoderTest.check_training_gradient_checkpointing with Musicgen->MusicgenMelody
    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.use_cache = False
        config.return_dict = True
        model = MusicgenMelodyForCausalLM(config)

        model.to(torch_device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        model.train()

        # Contrarily to the initial method, we don't unfreeze freezed parameters.
        # Indeed, sinusoidal position embeddings have frozen weights that should stay frozen.

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        inputs = self._prepare_for_class(inputs_dict, MusicgenMelodyForCausalLM, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()

        for k, v in model.named_parameters():
            if v.requires_grad:
                self.assertTrue(v.grad is not None, f"{k} in {MusicgenMelodyForCausalLM.__name__} has no gradient!")

    # override since we have to compute the input embeddings over codebooks
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]

            embed_tokens = model.get_input_embeddings()

            input_ids = input_ids.reshape(-1, config.num_codebooks, input_ids.shape[-1])

            inputs["inputs_embeds"] = sum(
                [embed_tokens[codebook](input_ids[:, codebook]) for codebook in range(config.num_codebooks)]
            )

            with torch.no_grad():
                model(**inputs)[0]

    # override since we have embeddings / LM heads over multiple codebooks
    def test_model_get_set_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            first_embed = model.get_input_embeddings()[0]
            self.assertIsInstance(first_embed, torch.nn.Embedding)
            lm_heads = model.get_output_embeddings()
            self.assertTrue(lm_heads is None or isinstance(lm_heads[0], torch.nn.Linear))

    @unittest.skip(reason="MusicGen melody does not use inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="this model doesn't support all arguments tested")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="this model has multiple inputs embeds and lm heads that should not be tied")
    def test_tie_model_weights(self):
        pass

    @unittest.skip(reason="this model has multiple inputs embeds and lm heads that should not be tied")
    def test_tied_weights_keys(self):
        pass

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {}
        return logits_processor_kwargs

    def test_greedy_generate_stereo_outputs(self):
        original_audio_channels = self.model_tester.audio_channels
        self.model_tester.audio_channels = 2
        super().test_greedy_generate_dict_outputs()
        self.model_tester.audio_channels = original_audio_channels

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.models.musicgen.test_modeling_musicgen.MusicgenDecoderTest.test_flash_attn_2_inference_equivalence
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, 1:] = 1
                    dummy_attention_mask[:, :1] = 0

                # Ignore copy
                outputs = model(dummy_input, output_hidden_states=True)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                # Ignore copy
                other_inputs = {
                    "output_hidden_states": True,
                }
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask

                outputs = model(dummy_input, **other_inputs)
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[1:], logits[1:], atol=4e-2, rtol=4e-2)

                # check with inference + dropout
                model.train()
                _ = model_fa(dummy_input, **other_inputs)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Copied from tests.models.musicgen.test_modeling_musicgen.MusicgenDecoderTest.test_flash_attn_2_inference_equivalence_right_padding
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, :-1] = 1
                    dummy_attention_mask[:, -1:] = 0

                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)

                    outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                else:
                    outputs = model(dummy_input, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)
                # Ignore copy
                other_inputs = {
                    "output_hidden_states": True,
                }
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask

                outputs = model(dummy_input, **other_inputs)
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[:-1], logits[:-1], atol=4e-2, rtol=4e-2)

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_eager_matches_sdpa_inference
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            model = model_class(config)

            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

                set_model_for_less_flaky_test(model_eager)
                set_model_for_less_flaky_test(model_sdpa)

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 8 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for batch_size in [7]:
                            # Ignore copy
                            batch_size_input_ids = self.model_tester.num_codebooks * batch_size
                            dummy_input = inputs_dict[model.main_input_name]

                            if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                dummy_input = dummy_input.to(torch_dtype)

                            # Ignore copy
                            dummy_input = dummy_input[:batch_size_input_ids]
                            # Ignore copy
                            if dummy_input.shape[0] != batch_size_input_ids:
                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    # Ignore copy
                                    extension = torch.rand(
                                        batch_size_input_ids - dummy_input.shape[0],
                                        *dummy_input.shape[1:],
                                        dtype=torch_dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                else:
                                    # Ignore copy
                                    extension = torch.randint(
                                        high=5,
                                        size=(batch_size_input_ids - dummy_input.shape[0], *dummy_input.shape[1:]),
                                        dtype=dummy_input.dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                            if not use_mask:
                                dummy_attention_mask = None
                            else:
                                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                if dummy_attention_mask is None:
                                    if is_encoder_decoder:
                                        seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                    else:
                                        seqlen = dummy_input.shape[-1]
                                    dummy_attention_mask = (
                                        torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                    )

                                dummy_attention_mask = dummy_attention_mask[:batch_size]
                                if dummy_attention_mask.shape[0] != batch_size:
                                    extension = torch.ones(
                                        batch_size - dummy_attention_mask.shape[0],
                                        *dummy_attention_mask.shape[1:],
                                        dtype=dummy_attention_mask.dtype,
                                        device=torch_device,
                                    )
                                    dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                    dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                dummy_attention_mask[:] = 1
                                if padding_side == "left":
                                    dummy_attention_mask[-1, :2] = 0
                                    dummy_attention_mask[-1, 2:] = 1
                                elif padding_side == "right":
                                    dummy_attention_mask[-1, -2:] = 0
                                    dummy_attention_mask[-1, :-2] = 1

                            for enable_kernels in [False, True]:
                                failcase = f"padding_side={padding_side}, use_mask={use_mask}, batch_size={batch_size}, enable_kernels={enable_kernels}"

                                other_inputs = {
                                    "output_hidden_states": True,
                                }

                                # Otherwise fails for e.g. WhisperEncoderModel
                                if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                                    other_inputs["attention_mask"] = dummy_attention_mask

                                # TODO: test gradients as well (& for FA2 as well!)
                                with torch.no_grad():
                                    with sdpa_kernel(
                                        enable_flash=enable_kernels,
                                        enable_math=True,
                                        enable_mem_efficient=enable_kernels,
                                    ):
                                        outputs_eager = model_eager(dummy_input, **other_inputs)
                                        outputs_sdpa = model_sdpa(dummy_input, **other_inputs)

                                logits_eager = (
                                    outputs_eager.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_eager.decoder_hidden_states[-1]
                                )
                                logits_sdpa = (
                                    outputs_sdpa.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_sdpa.decoder_hidden_states[-1]
                                )

                                if torch_device in ["cpu", "cuda"]:
                                    atol = atols[torch_device, enable_kernels, torch_dtype]
                                    rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                elif torch_device == "xpu":
                                    # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                                    # which is implemented on PyTorch level using aten operators and is
                                    # device agnostic with respect to implementation of each aten operator.
                                    atol = atols["cuda", False, torch_dtype]
                                    rtol = rtols["cuda", False, torch_dtype]
                                else:
                                    atol = 1e-7
                                    rtol = 1e-4

                                # Masked tokens output slightly deviates - we don't mind that.
                                if use_mask:
                                    _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                                    _logits_eager = torch.zeros_like(input=logits_eager)

                                    _logits_sdpa[:-1] = logits_sdpa[:-1]
                                    _logits_eager[:-1] = logits_eager[:-1]

                                    if padding_side == "left":
                                        _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                                        _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                                    elif padding_side == "right":
                                        _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                                        _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                                    logits_sdpa = _logits_sdpa
                                    logits_eager = _logits_eager

                                results = [
                                    torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                                    for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
                                ]
                                # If 80% batch elements have matched results, it's fine
                                if np.mean(results) < 0.8:
                                    fail_cases.append(
                                        get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                    )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))


def prepare_musicgen_melody_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    labels=None,
):
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.reshape(
            -1, config.decoder.num_codebooks, decoder_input_ids.shape[-1]
        )[:, 0, :]
        decoder_attention_mask = decoder_attention_mask.ne(config.decoder.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(
            config.text_encoder.num_hidden_layers, config.text_encoder.num_attention_heads, device=torch_device
        )
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "labels": labels,
    }


class MusicgenMelodyTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers because of #29297
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
        num_filters=4,
        codebook_size=128,
        conditional_seq_length=3,
        chroma_length=24,
        audio_channels=1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.num_filters = num_filters
        self.codebook_size = codebook_size
        self.conditional_seq_length = conditional_seq_length
        self.chroma_length = chroma_length
        self.encoder_seq_length = conditional_seq_length + seq_length
        self.audio_channels = audio_channels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.conditional_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_musicgen_melody_inputs_dict(config, input_ids, decoder_input_ids=decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        text_encoder_config = T5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.intermediate_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
        )
        audio_encoder_config = EncodecConfig(
            hidden_size=self.vocab_size,
            compress=1,
            num_filters=self.num_filters,
            codebook_size=self.codebook_size,
            codebook_dim=self.vocab_size,
        )
        decoder_config = MusicgenMelodyDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
            audio_channels=self.audio_channels,
        )
        config = MusicgenMelodyConfig.from_sub_models_config(
            text_encoder_config, audio_encoder_config, decoder_config, chroma_length=self.chroma_length
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
# Copied from tests.models.musicgen.test_modeling_musicgen.MusicgenTest with Musicgen->MusicgenMelody, musicgen->musicgen_melody, EncoderDecoder->DecoderOnly, input_values->input_features
class MusicgenMelodyTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenMelodyForConditionalGeneration,) if is_torch_available() else ()
    greedy_sample_model_classes = (MusicgenMelodyForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-to-audio": MusicgenMelodyForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False  # training is not supported yet for MusicGen
    test_headmasking = False
    test_resize_embeddings = False
    # not to test torchscript as the model tester doesn't prepare `input_features` and `padding_mask`
    # (and `torchscript` hates `None` values).
    test_torchscript = False
    _is_composite = True

    def setUp(self):
        self.model_tester = MusicgenMelodyTester(self)

    # special case for labels
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_codebooks),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True
            model = model_class(config)

            model.to(torch_device)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            model.train()

            # The audio encoder weights are not used during the forward pass (only during the generate pass)
            # So we need to freeze it to be able to train.
            model.freeze_audio_encoder()

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()

            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.assertTrue(v.grad is not None, f"{k} in {model_class.__name__} has no gradient!")

    # Ignore copy
    def _check_output_with_attentions(self, outputs, config, input_ids, decoder_input_ids):
        decoder_config = config.decoder

        decoder_attentions = outputs["attentions"]
        num_decoder_layers = decoder_config.num_hidden_layers
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        output_shape = decoder_input_ids.shape[-1] + input_ids.shape[-1] + self.model_tester.chroma_length
        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, output_shape, output_shape),
        )

    def check_musicgen_melody_model_output_attentions(
        self,
        model_class,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=True,
                **kwargs,
            )
        self._check_output_with_attentions(outputs, config, input_ids, decoder_input_ids)

    # Ignore copy
    def check_musicgen_melody_model_output_attentions_from_config(
        self,
        model_class,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        # Similar to `check_musicgen_melody_model_output_attentions`, but with `output_attentions` triggered from the
        # config file. Contrarily to most models, changing the model's config won't work -- the defaults are loaded
        # from the inner models' configurations.
        config.output_attentions = True  # model config -> won't work

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        self.assertTrue(all(key not in outputs for key in ["encoder_attentions", "decoder_attentions"]))
        config.text_encoder.output_attentions = True  # inner model config -> will work
        config.audio_encoder.output_attentions = True
        config.decoder.output_attentions = True

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        self._check_output_with_attentions(outputs, config, input_ids, decoder_input_ids)

    # override since changing `output_attentions` from the top-level model config won't work
    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            self.check_musicgen_melody_model_output_attentions(model_class, config, **inputs_dict)
            self.check_musicgen_melody_model_output_attentions_from_config(model_class, config, **inputs_dict)

    # override since we have a specific forward signature for musicgen_melody
    # Ignore copy
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_ids",
                "attention_mask",
                "input_features",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            if "head_mask" and "decoder_head_mask" in arg_names:
                expected_arg_names.extend(["head_mask", "decoder_head_mask"])

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # override since changing `gradient_checkpointing` from the top-level model config won't work
    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.text_encoder.gradient_checkpointing = True
            config.audio_encoder.gradient_checkpointing = True
            config.decoder.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    @unittest.skip(reason="MusicGen has multiple inputs embeds and lm heads that should not be tied.")
    def test_tie_model_weights(self):
        pass

    @unittest.skip(reason="MusicGen has multiple inputs embeds and lm heads that should not be tied.")
    def test_tied_model_weights_key_ignore(self):
        pass

    @unittest.skip(reason="MusicGen has multiple inputs embeds and lm heads that should not be tied.")
    def test_tied_weights_keys(self):
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

    # override since changing `output_hidden_states` / `output_attentions` from the top-level model config won't work
    # Ignore copy
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_encoder.output_hidden_states = True
        config.audio_encoder.output_hidden_states = True
        config.decoder.output_hidden_states = True

        config.text_encoder.output_attentions = True
        config.decoder.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states
        encoder_hidden_states.retain_grad()

        decoder_hidden_states = outputs.hidden_states[0]
        decoder_hidden_states.retain_grad()

        if self.has_attentions:
            decoder_attentions = outputs.attentions[0]
            decoder_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(decoder_hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(decoder_attentions.grad)

    # override since changing `output_hidden_states` from the top-level model config won't work
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states

            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            # Ignore copy
            seq_length = self.model_tester.conditional_seq_length + self.model_tester.chroma_length
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            # Ignore copy
            seq_length = self.model_tester.encoder_seq_length + self.model_tester.chroma_length
            # Ignore copy
            expected_num_layers = self.model_tester.num_hidden_layers + 1
            # Ignore copy
            hidden_states = outputs.hidden_states
            self.assertIsInstance(hidden_states, (list, tuple))
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.text_encoder.output_hidden_states = True
            config.audio_encoder.output_hidden_states = True
            config.decoder.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # override since the conv layers and lstm's in encodec are exceptions
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
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

    # override since we have embeddings / LM heads over multiple codebooks
    def test_model_get_set_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), torch.nn.Embedding)
            lm_heads = model.get_output_embeddings()
            self.assertTrue(lm_heads is None or isinstance(lm_heads[0], torch.nn.Linear))

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {}
        return logits_processor_kwargs

    @require_torch_fp16
    @require_torch_accelerator  # not all operations are supported in fp16 on CPU
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()

        for model_class in self.greedy_sample_model_classes:
            model = model_class(config).eval().to(torch_device)
            model.half()
            # greedy
            model.generate(input_dict["input_ids"], attention_mask=input_dict["attention_mask"], max_new_tokens=10)
            # sampling
            model.generate(
                input_dict["input_ids"], attention_mask=input_dict["attention_mask"], do_sample=True, max_new_tokens=10
            )

    def test_greedy_generate_stereo_outputs(self):
        original_audio_channels = self.model_tester.audio_channels
        self.model_tester.audio_channels = 2
        super().test_greedy_generate_dict_outputs()
        self.model_tester.audio_channels = original_audio_channels

    @unittest.skip(
        reason="MusicgenMelodyModel is actually not the base of MusicgenMelodyForCausalLM as the latter is a composit model"
    )
    def test_save_load_fast_init_from_base(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Adapted from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_inference_equivalence
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.bfloat16,
                    attn_implementation={"decoder": "flash_attention_2", "audio_encoder": None, "text_encoder": None},
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, 1:] = 1
                    dummy_attention_mask[:, :1] = 0

                # Ignore copy
                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)
                # Ignore copy
                outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)
                # Ignore copy
                other_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": dummy_attention_mask,
                    "output_hidden_states": True,
                }
                # Ignore copy
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask
                # Ignore copy
                outputs = model(dummy_input, **other_inputs)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[1:], logits[1:], atol=4e-2, rtol=4e-2)

                # check with inference + dropout
                model.train()
                _ = model_fa(dummy_input, **other_inputs)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_conversion(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation={"decoder": "flash_attention_2", "audio_encoder": None, "text_encoder": None},
                ).to(torch_device)

                for _, module in model.named_modules():
                    if "FlashAttention" in module.__class__.__name__:
                        return

                self.assertTrue(False, "FlashAttention2 modules not found in model")

    @require_torch_sdpa
    @require_torch_gpu
    @slow
    def test_sdpa_can_dispatch_on_flash(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        torch.compiler.reset()
        compute_capability = torch.cuda.get_device_capability()
        major, _ = compute_capability

        if not torch.version.cuda or major < 8:
            self.skipTest(reason="This test requires an NVIDIA GPU with compute capability >= 8.0")

        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            if config.model_type in ["llava", "llava_next", "vipllava", "video_llava"]:
                self.skipTest(
                    reason="Llava-like models currently (transformers==4.39.1) requires an attention_mask input"
                )
            if config.model_type in ["paligemma"]:
                self.skipTest(
                    "PaliGemma-like models currently (transformers==4.41.0) requires an attention_mask input"
                )
            if config.model_type in ["idefics", "idefics2", "idefics3"]:
                self.skipTest(reason="Idefics currently (transformers==4.39.1) requires an image_attention_mask input")
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation={"decoder": "sdpa", "audio_encoder": None, "text_encoder": None},
                )
                model.to(torch_device)

                inputs_dict.pop("attention_mask", None)
                inputs_dict.pop("decoder_attention_mask", None)

                for name, inp in inputs_dict.items():
                    if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float32, torch.float16]:
                        inputs_dict[name] = inp.to(torch.float16)

                with sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    _ = model(**inputs_dict)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    # Adapted from tests.test_modeling_common.ModelTesterMixin.test_flash_attn_2_inference_equivalence_right_padding
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.bfloat16,
                    attn_implementation={"decoder": "flash_attention_2", "audio_encoder": None, "text_encoder": None},
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                # Ignore copy
                dummy_input = inputs_dict[model.main_input_name]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    # Ignore copy
                    dummy_attention_mask[:, :-1] = 1
                    dummy_attention_mask[:, -1:] = 0

                # Ignore copy
                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)
                # Ignore copy
                outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                # Ignore copy
                other_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": dummy_attention_mask,
                    "output_hidden_states": True,
                }
                # Ignore copy
                if dummy_attention_mask is not None:
                    other_inputs["attention_mask"] = dummy_attention_mask
                # Ignore copy
                outputs = model(dummy_input, **other_inputs)
                # Ignore copy
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa[:-1], logits[:-1], atol=4e-2, rtol=4e-2)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                audio_encoder_attn = "sdpa" if model.audio_encoder._supports_sdpa else "eager"
                text_encoder_attn = "sdpa" if model.text_encoder._supports_sdpa else "eager"
                decoder_attn = "sdpa" if model.decoder._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.audio_encoder.config._attn_implementation == audio_encoder_attn)
                self.assertTrue(model_sdpa.text_encoder.config._attn_implementation == text_encoder_attn)
                self.assertTrue(model_sdpa.decoder.config._attn_implementation == decoder_attn)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)

                self.assertTrue(model_eager.audio_encoder.config._attn_implementation == "eager")
                self.assertTrue(model_eager.text_encoder.config._attn_implementation == "eager")
                self.assertTrue(model_eager.decoder.config._attn_implementation == "eager")
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    if "SdpaAttention" in submodule.__class__.__name__:
                        has_sdpa = True
                        break
                if not has_sdpa and model_sdpa.config.model_type != "falcon":
                    raise ValueError("The SDPA model should have SDPA attention layers")

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    # Copied from tests.test_modeling_common.ModelTesterMixin.test_eager_matches_sdpa_inference
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            model = model_class(config)

            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

                set_model_for_less_flaky_test(model_eager)
                set_model_for_less_flaky_test(model_sdpa)

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 8 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for batch_size in [7]:
                            dummy_input = inputs_dict[model.main_input_name]

                            if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                dummy_input = dummy_input.to(torch_dtype)

                            dummy_input = dummy_input[:batch_size]
                            if dummy_input.shape[0] != batch_size:
                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    extension = torch.rand(
                                        batch_size - dummy_input.shape[0],
                                        *dummy_input.shape[1:],
                                        dtype=torch_dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                else:
                                    extension = torch.randint(
                                        high=5,
                                        size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                        dtype=dummy_input.dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                            if not use_mask:
                                dummy_attention_mask = None
                            else:
                                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                if dummy_attention_mask is None:
                                    # Ignore copy
                                    seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                    # Ignore copy
                                    dummy_attention_mask = (
                                        torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                    )

                                dummy_attention_mask = dummy_attention_mask[:batch_size]
                                if dummy_attention_mask.shape[0] != batch_size:
                                    extension = torch.ones(
                                        batch_size - dummy_attention_mask.shape[0],
                                        *dummy_attention_mask.shape[1:],
                                        dtype=dummy_attention_mask.dtype,
                                        device=torch_device,
                                    )
                                    dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                    dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                dummy_attention_mask[:] = 1
                                if padding_side == "left":
                                    dummy_attention_mask[-1, :2] = 0
                                    dummy_attention_mask[-1, 2:] = 1
                                elif padding_side == "right":
                                    dummy_attention_mask[-1, -2:] = 0
                                    dummy_attention_mask[-1, :-2] = 1

                            for enable_kernels in [False, True]:
                                failcase = f"padding_side={padding_side}, use_mask={use_mask}, batch_size={batch_size}, enable_kernels={enable_kernels}"
                                # Ignore copy
                                batch_size_input_ids = self.model_tester.num_codebooks * batch_size
                                # Ignore copy
                                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[
                                    :batch_size_input_ids
                                ]
                                # Ignore copy
                                if decoder_input_ids.shape[0] != batch_size_input_ids:
                                    # Ignore copy
                                    extension = torch.ones(
                                        batch_size_input_ids - decoder_input_ids.shape[0],
                                        *decoder_input_ids.shape[1:],
                                        dtype=decoder_input_ids.dtype,
                                        device=torch_device,
                                    )
                                    decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                                    decoder_input_ids = decoder_input_ids.to(torch_device)

                                # TODO: never an `attention_mask` arg here?
                                # Ignore copy
                                other_inputs = {
                                    "decoder_input_ids": decoder_input_ids,
                                    "decoder_attention_mask": dummy_attention_mask,
                                    "output_hidden_states": True,
                                }

                                # TODO: test gradients as well (& for FA2 as well!)
                                # Ignore copy
                                with torch.no_grad():
                                    with sdpa_kernel(
                                        enable_flash=enable_kernels,
                                        enable_math=True,
                                        enable_mem_efficient=enable_kernels,
                                    ):
                                        outputs_eager = model_eager(dummy_input, **other_inputs)
                                        outputs_sdpa = model_sdpa(dummy_input, **other_inputs)

                                logits_eager = (
                                    outputs_eager.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_eager.decoder_hidden_states[-1]
                                )
                                logits_sdpa = (
                                    outputs_sdpa.hidden_states[-1]
                                    if not is_encoder_decoder
                                    else outputs_sdpa.decoder_hidden_states[-1]
                                )

                                if torch_device in ["cpu", "cuda"]:
                                    atol = atols[torch_device, enable_kernels, torch_dtype]
                                    rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                elif torch_device == "xpu":
                                    # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                                    # which is implemented on PyTorch level using aten operators and is
                                    # device agnostic with respect to implementation of each aten operator.
                                    atol = atols["cuda", False, torch_dtype]
                                    rtol = rtols["cuda", False, torch_dtype]
                                else:
                                    atol = 1e-7
                                    rtol = 1e-4

                                # Masked tokens output slightly deviates - we don't mind that.
                                if use_mask:
                                    _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                                    _logits_eager = torch.zeros_like(input=logits_eager)

                                    _logits_sdpa[:-1] = logits_sdpa[:-1]
                                    _logits_eager[:-1] = logits_eager[:-1]

                                    if padding_side == "left":
                                        _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                                        _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                                    elif padding_side == "right":
                                        _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                                        _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                                    logits_sdpa = _logits_sdpa
                                    logits_eager = _logits_eager

                                results = [
                                    torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                                    for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
                                ]
                                # If 80% batch elements have matched results, it's fine
                                if np.mean(results) < 0.8:
                                    fail_cases.append(
                                        get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                    )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    def test_requires_grad_with_frozen_encoders(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.freeze_audio_encoder()

            audio_encoder_grads = [param.requires_grad for param in model.audio_encoder.parameters()]
            text_encoder_grads = [param.requires_grad for param in model.text_encoder.parameters()]

            self.assertFalse(all(audio_encoder_grads))
            self.assertTrue(all(text_encoder_grads))

            model = model_class(config)
            model.freeze_text_encoder()

            audio_encoder_grads = [param.requires_grad for param in model.audio_encoder.parameters()]
            text_encoder_grads = [param.requires_grad for param in model.text_encoder.parameters()]

            self.assertTrue(all(audio_encoder_grads))
            self.assertFalse(all(text_encoder_grads))


# Copied from tests.models.musicgen.test_modeling_musicgen.get_bip_bip
def get_bip_bip(bip_duration=0.125, duration=0.5, sample_rate=32000):
    """Produces a series of 'bip bip' sounds at a given frequency."""
    timesteps = np.arange(int(duration * sample_rate)) / sample_rate
    wav = np.cos(2 * math.pi * 440 * timesteps)
    time_period = (timesteps % (2 * bip_duration)) / (2 * bip_duration)
    envelope = time_period >= 0.5
    return wav * envelope


@require_torch
@require_torchaudio
class MusicgenMelodyIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return MusicgenMelodyForConditionalGeneration.from_pretrained("ylacombe/musicgen-melody").to(torch_device)

    @cached_property
    def processor(self):
        return MusicgenMelodyProcessor.from_pretrained("ylacombe/musicgen-melody")

    @slow
    def test_logits_text_prompt(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        # prepare the decoder inputs
        pad_token_id = model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones((input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long).to(torch_device)
            * pad_token_id
        )

        with torch.no_grad():
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
            1.1100, -2.1065, -3.7699, -0.7102,  1.3707, -1.7028, -2.6802, -6.0367,
            1.0504, -2.5358, -4.3497,  0.7338,  0.4823, -2.5260,  1.2717,  1.5427
        ])
        # fmt: on
        EXPECTED_OUTPUT_LENGTH = input_ids.shape[1] + 1 + self.model.config.chroma_length

        logits_shape = (
            input_ids.shape[0] * model.decoder.num_codebooks,
            EXPECTED_OUTPUT_LENGTH,
            model.decoder.config.vocab_size,
        )

        self.assertTrue(logits.shape == logits_shape)
        self.assertTrue(torch.allclose(logits[0, -1, :16].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_logits_text_audio_prompt(self):
        model = self.model
        processor = self.processor

        audio = [get_bip_bip(duration=0.5), get_bip_bip(duration=1.0)]
        text = ["80s music", "Club techno"]

        inputs = processor(audio=audio, text=text, padding=True, return_tensors="pt")

        # prepare the text encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        # prepare the audio encoder inputs
        input_features = inputs.input_features.to(torch_device)

        # prepare the decoder inputs
        pad_token_id = model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones((input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long).to(torch_device)
            * pad_token_id
        )

        with torch.no_grad():
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
            ).logits

        # fmt: off
        EXPECTED_LOGITS = torch.tensor([
        [ 0.7479,  0.3742,  0.6253, -7.9405,  0.7105, -6.9995,  0.7792, -3.0482],
        [-2.7905,  0.7492, -0.2556, -8.1586, -1.6740,  0.5771, -8.3650, -0.0908]
        ])
        # fmt: on

        self.assertTrue(logits.shape == (8, 240, 2048))
        self.assertTrue(torch.allclose(logits[1:3, -1, 32:40].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_generate_unconditional_greedy(self):
        model = self.model

        # only generate 1 sample with greedy - since it's deterministic all elements of the batch will be the same
        unconditional_inputs = self.processor.get_unconditional_inputs(num_samples=1).to(torch_device)

        output_values = model.generate(**unconditional_inputs, do_sample=False, max_new_tokens=10, guidance_scale=1.0)

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                1.2741e-04, -8.0466e-05,  5.5789e-04,  1.0402e-03,  2.6547e-04,
                1.5587e-05, -1.4210e-04, -9.7303e-05,  6.4504e-04,  5.0903e-04,
                9.6474e-04,  1.0498e-03,  3.7210e-05, -5.3652e-04, -3.6579e-04, -2.5678e-04
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (1, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_unconditional_sampling(self):
        model = self.model

        # for stochastic sampling we can generate multiple outputs
        unconditional_inputs = self.processor.get_unconditional_inputs(num_samples=2).to(torch_device)

        set_seed(0)

        output_values = model.generate(
            **unconditional_inputs, do_sample=True, max_new_tokens=10, guidance_scale=1.0, temperature=1.0, top_k=250
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -0.0085, -0.0160,  0.0028,  0.0005, -0.0095,  0.0028, -0.0122, -0.0299,
                -0.0052, -0.0145,  0.0092,  0.0063, -0.0378, -0.0621, -0.0784, -0.0120,
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_prompt_greedy(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        output_values = model.generate(
            input_ids, attention_mask=attention_mask, do_sample=False, guidance_scale=None, max_new_tokens=10
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                1.2741e-04, -8.0474e-05,  5.5789e-04,  1.0402e-03,  2.6547e-04,
                1.5597e-05, -1.4210e-04, -9.7309e-05,  6.4504e-04,  5.0903e-04
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :10].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_prompt_greedy_with_classifier_free_guidance(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        output_values = model.generate(
            input_ids, attention_mask=attention_mask, do_sample=False, guidance_scale=3, max_new_tokens=10
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                1.2741e-04, -8.0474e-05,  5.5789e-04,  1.0402e-03,  2.6547e-04,
                1.5597e-05, -1.4210e-04, -9.7309e-05,  6.4504e-04,  5.0903e-04,
                9.6475e-04,  1.0499e-03,  3.7215e-05, -5.3651e-04, -3.6578e-04, -2.5678e-04
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_prompt_sampling(self):
        model = self.model
        processor = self.processor

        inputs = processor(text=["80s music", "Club techno"], padding=True, return_tensors="pt")

        # prepare the encoder inputs
        input_ids = inputs.input_ids.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        set_seed(0)
        output_values = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            guidance_scale=None,
            max_new_tokens=10,
            temperature=1.0,
            top_k=250,
        )

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -0.0165, -0.0222, -0.0041, -0.0058, -0.0145, -0.0023, -0.0160, -0.0310,
                -0.0055, -0.0127,  0.0104,  0.0105, -0.0326, -0.0611, -0.0744, -0.0083
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))

    @slow
    def test_generate_text_audio_prompt(self):
        model = self.model
        processor = self.processor

        audio = [get_bip_bip(duration=0.5), get_bip_bip(duration=1.0)]
        text = ["80s music", "Club techno"]

        inputs = processor(audio=audio, text=text, padding=True, return_tensors="pt").to(torch_device)

        output_values = model.generate(**inputs, do_sample=False, guidance_scale=None, max_new_tokens=10)

        # fmt: off
        EXPECTED_VALUES = torch.tensor(
            [
                -1.1999e-04, -2.2303e-04,  4.6296e-04,  1.0524e-03,  2.4827e-04,
                -4.0294e-05, -1.2468e-04,  4.9846e-05,  7.1484e-04,  4.4198e-04,
                7.9063e-04,  8.8141e-04, -6.1807e-05, -6.1856e-04, -3.6235e-04, -2.7226e-04
            ]
        )
        # fmt: on

        self.assertTrue(output_values.shape == (2, 1, 4480))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES, atol=1e-4))


@require_torch
@require_torchaudio
class MusicgenMelodyStereoIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return MusicgenMelodyForConditionalGeneration.from_pretrained("ylacombe/musicgen-stereo-melody").to(
            torch_device
        )

    @cached_property
    def processor(self):
        return MusicgenMelodyProcessor.from_pretrained("ylacombe/musicgen-stereo-melody")

    @slow
    def test_generate_unconditional_greedy(self):
        model = self.model

        # only generate 1 sample with greedy - since it's deterministic all elements of the batch will be the same
        unconditional_inputs = self.processor.get_unconditional_inputs(num_samples=1).to(torch_device)

        output_values = model.generate(**unconditional_inputs, do_sample=False, max_new_tokens=12, guidance_scale=1.0)

        # fmt: off
        EXPECTED_VALUES_LEFT = torch.tensor(
            [
                1.2742e-04, -8.0480e-05,  5.5788e-04,  1.0401e-03,  2.6547e-04,
                1.5587e-05, -1.4211e-04, -9.7308e-05,  6.4503e-04,  5.0903e-04,
                9.6475e-04,  1.0499e-03,  3.7205e-05, -5.3652e-04, -3.6579e-04, 2.5679e-04
            ]
        )
        # fmt: on

        # (bsz, channels, seq_len)
        self.assertTrue(output_values.shape == (1, 2, 5760))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES_LEFT, atol=6e-4))
        self.assertTrue(torch.allclose(output_values[0, 1, :16].cpu(), EXPECTED_VALUES_LEFT, atol=6e-4))

    @slow
    def test_generate_text_audio_prompt(self):
        model = self.model
        processor = self.processor

        audio = [get_bip_bip(duration=0.5), get_bip_bip(duration=1.0)]
        text = ["80s music", "Club techno"]

        inputs = processor(audio=audio, text=text, padding=True, return_tensors="pt").to(torch_device)

        output_values = model.generate(**inputs, do_sample=False, guidance_scale=3.0, max_new_tokens=12)

        # fmt: off
        EXPECTED_VALUES_LEFT_FIRST_SAMPLE = torch.tensor(
            [
                -0.0862, -0.1021, -0.0936, -0.0754, -0.0616, -0.0456, -0.0354, -0.0298,
                -0.0036,  0.0222,  0.0523,  0.0660,  0.0496,  0.0356,  0.0457,  0.0769
            ]
        )
        EXPECTED_VALUES_RIGHT_SECOND_SAMPLE = torch.tensor(
            [
                -0.0327, -0.0450, -0.0264, -0.0278, -0.0365, -0.0272, -0.0401, -0.0574,
                -0.0413, -0.0508, -0.0269, -0.0323, -0.0762, -0.1115, -0.1390, -0.0790
            ]
        )
        # fmt: on

        # (bsz, channels, seq_len)
        self.assertTrue(output_values.shape == (2, 2, 5760))
        self.assertTrue(torch.allclose(output_values[0, 0, :16].cpu(), EXPECTED_VALUES_LEFT_FIRST_SAMPLE, atol=1e-4))
        self.assertTrue(torch.allclose(output_values[1, 1, :16].cpu(), EXPECTED_VALUES_RIGHT_SECOND_SAMPLE, atol=1e-4))
