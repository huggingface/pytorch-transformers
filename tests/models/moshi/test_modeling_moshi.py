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
"""Testing suite for the PyTorch Moshi model."""

import copy
import tempfile
import unittest

import numpy as np
import pytest
from datasets import Audio, load_dataset
from parameterized import parameterized

from transformers import (
    MoshiConfig,
    PretrainedConfig,
)
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
)
from transformers.testing_utils import (
    is_flaky,
    is_torch_available,
    require_torch,
    require_torch_fp16,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_deepspeed_available():
    import deepspeed

if is_torch_available():
    import torch

    from transformers import (
        AutoFeatureExtractor,
        AutoTokenizer,
        MoshiForCausalLM,
        MoshiForConditionalGeneration,
        MoshiModel,
    )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


class MoshiDecoderTester:
    def __init__(
        self,
        parent,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="silu",
        rms_norm_eps=0.001,
        ffn_dim=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=25,
        num_codebooks=4,
        audio_encoder_type="mimi",
        attn_implementation="eager",
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
        self.rms_norm_eps = rms_norm_eps
        self.ffn_dim = ffn_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.num_codebooks = num_codebooks
        self.audio_encoder_type = audio_encoder_type
        self.attn_implementation = attn_implementation

    def prepare_config_and_inputs(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        input_ids = ids_tensor([batch_size, self.seq_length], self.vocab_size)
        config = self.get_config()

        attention_mask = input_ids.ne(self.pad_token_id)

        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def get_config(self):
        config = MoshiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            num_codebooks=self.num_codebooks,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=False,
            pad_token_id=self.pad_token_id,
            ffn_dim=self.ffn_dim,
            audio_encoder_config={"model_type": self.audio_encoder_type},
            attn_implementation=self.attn_implementation,
        )
        return config

    def prepare_config_and_inputs_for_common(self, batch_size=None):
        config, inputs_dict = self.prepare_config_and_inputs(batch_size)
        return config, inputs_dict


@require_torch
class MoshiDecoderTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MoshiModel, MoshiForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (
        (MoshiForCausalLM,) if is_torch_available() else ()
    )  # we don't want to run all the generation tests, only a specific subset
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False
    pipeline_model_mapping = (
        {
            "feature-extraction": MoshiModel,
            "text-generation": MoshiForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MoshiDecoderTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=MoshiConfig,
            hidden_size=16,
            audio_encoder_config={"model_type": self.model_tester.audio_encoder_type},
        )

    @unittest.skip(reason="The MoshiModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    def _get_input_ids_and_config(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(batch_size)
        input_ids = inputs_dict.pop("input_ids").to(torch_device)
        attention_mask = inputs_dict.pop("attention_mask").to(torch_device)

        return config, input_ids, attention_mask, inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {}
        return logits_processor_kwargs

    @require_torch_sdpa
    @slow
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        self.skipTest(reason="Moshi has no strict equivalence between two modes, skipping this test.")

    # Copied from tests.test_modeling_common.ModelTesterMixin.test_resize_tokens_embeddings
    def test_resize_tokens_embeddings(self):
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to `False`")

        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_embed_pre_resize = model.get_input_embeddings()
            type_model_embed_pre_resize = type(model_embed_pre_resize)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check to make sure the type of embeddings returned post resizing is same as type of input
            type_model_embed_post_resize = type(model_embed)
            self.assertEqual(type_model_embed_pre_resize, type_model_embed_post_resize)
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(model_embed.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, rtol=1e-3, atol=1e-3)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            # make sure that decoder_input_ids are resized as well
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                if "decoder_input_ids" in inputs_dict:
                    inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
                model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            del model
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_vocab_size = config.get_text_config().vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertTrue(new_model_vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], new_model_vocab_size)
            self.assertTrue(new_model_vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

            # Test when `vocab_size` is smaller than `hidden_size`.
            del model
            config.vocab_size = 4
            config.pad_token_id = 4  # Ignore copy
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check to make sure the type of embeddings returned post resizing is same as type of input
            type_model_embed_post_resize = type(model_embed)
            self.assertEqual(type_model_embed_pre_resize, type_model_embed_post_resize)
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(model_embed.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, rtol=1e-3, atol=1e-3)

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @is_flaky(max_attempts=5, description="flaky on some models.")
    def test_save_load(self):
        super().test_save_load()


class MoshiTester:
    def __init__(
        self,
        parent,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=4,
        hidden_act="silu",
        rms_norm_eps=0.001,
        ffn_dim=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=25,
        bos_token_id=25,
        num_codebooks=4,
        audio_encoder_type="mimi",
        attn_implementation="eager",
        depth_hidden_size=16,
        depth_num_hidden_layers=2,
        depth_max_position_embeddings=5,
        depth_num_attention_heads=8,
        depth_ffn_dim=16,
        depth_sliding_window=4,
        mimi_intermediate_size=40,
        mimi_hidden_size=32,
        mimi_num_filters=8,
        mimi_num_residual_layers=1,
        mimi_upsampling_ratios=[8, 4],
        mimi_codebook_size=64,
        mimi_vector_quantization_hidden_dimension=64,
        mimi_codebook_dim=64,
        mimi_upsample_groups=32,
        mimi_num_hidden_layers=2,
        mimi_num_attention_heads=2,
        mimi_num_key_value_heads=2,
        mimi_sliding_window=3,
        sampling_rate=800,
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
        self.rms_norm_eps = rms_norm_eps
        self.ffn_dim = ffn_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.attn_implementation = attn_implementation
        self.depth_hidden_size = depth_hidden_size
        self.depth_num_hidden_layers = depth_num_hidden_layers
        self.depth_max_position_embeddings = depth_max_position_embeddings
        self.depth_num_attention_heads = depth_num_attention_heads
        self.depth_ffn_dim = depth_ffn_dim
        self.depth_sliding_window = depth_sliding_window

        self.audio_encoder_type = audio_encoder_type
        self.mimi_intermediate_size = mimi_intermediate_size
        self.mimi_hidden_size = mimi_hidden_size
        self.mimi_num_filters = mimi_num_filters
        self.mimi_num_residual_layers = mimi_num_residual_layers
        self.mimi_upsampling_ratios = mimi_upsampling_ratios
        self.mimi_codebook_size = mimi_codebook_size
        self.mimi_vector_quantization_hidden_dimension = mimi_vector_quantization_hidden_dimension
        self.mimi_codebook_dim = mimi_codebook_dim
        self.mimi_upsample_groups = mimi_upsample_groups
        self.mimi_num_hidden_layers = mimi_num_hidden_layers
        self.mimi_num_attention_heads = mimi_num_attention_heads
        self.mimi_num_key_value_heads = mimi_num_key_value_heads
        self.mimi_sliding_window = mimi_sliding_window
        self.sampling_rate = sampling_rate

        self.num_hidden_states_types = 2

    def prepare_config_and_inputs(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        input_ids = ids_tensor([batch_size, self.seq_length], self.vocab_size)

        moshi_audio_codes = ids_tensor([batch_size, self.num_codebooks, self.seq_length], self.mimi_codebook_size)
        user_audio_codes = ids_tensor([batch_size, self.num_codebooks, self.seq_length], self.mimi_codebook_size)
        attention_mask = input_ids.ne(self.pad_token_id)

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "moshi_audio_codes": moshi_audio_codes,
            "user_audio_codes": user_audio_codes,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def get_config(self):
        mimi_dict_config = {
            "model_type": self.audio_encoder_type,
            "audio_channels": 1,
            "hidden_size": self.mimi_hidden_size,
            "num_filters": self.mimi_num_filters,
            "num_residual_layers": self.mimi_num_residual_layers,
            "upsampling_ratios": self.mimi_upsampling_ratios,
            "codebook_size": self.mimi_codebook_size,
            "vector_quantization_hidden_dimension": self.mimi_vector_quantization_hidden_dimension,
            "upsample_groups": self.mimi_upsample_groups,
            "num_hidden_layers": self.mimi_num_hidden_layers,
            "num_attention_heads": self.mimi_num_attention_heads,
            "num_key_value_heads": self.mimi_num_key_value_heads,
            "sliding_window": self.mimi_sliding_window,
            "codebook_dim": self.mimi_codebook_dim,
            "use_cache": False,
            "sampling_rate": self.sampling_rate,
        }

        depth_dict_config = {
            "hidden_size": self.depth_hidden_size,
            "num_hidden_layers": self.depth_num_hidden_layers,
            "max_position_embeddings": self.depth_max_position_embeddings,
            "num_attention_heads": self.depth_num_attention_heads,
            "ffn_dim": self.depth_ffn_dim,
            "sliding_window": self.depth_sliding_window,
        }

        config = MoshiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            num_codebooks=self.num_codebooks,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            ffn_dim=self.ffn_dim,
            audio_encoder_config=mimi_dict_config,
            depth_decoder_config=depth_dict_config,
            attn_implementation=self.attn_implementation,
        )
        return config

    def prepare_config_and_inputs_for_common(self, batch_size=None):
        config, inputs_dict = self.prepare_config_and_inputs(batch_size)
        return config, inputs_dict


@require_torch
class MoshiTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MoshiForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (MoshiForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False  # training is not supported yet for Moshi
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = MoshiTester(self)

    # special case for labels
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            inputs_dict["text_labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(batch_size)
        input_ids = inputs_dict.pop("input_ids").to(torch_device)
        attention_mask = inputs_dict.pop("attention_mask").to(torch_device)

        # Make sure we only return `input_ids`.
        # Note that audio_codes will still be generated internally, so the ability to test audio codes is still there.
        # There are further tests to test that audio waveforms and codes are well generated.
        inputs_dict["return_audio_waveforms"] = False
        inputs_dict["return_audio_codes"] = False
        inputs_dict["concat_unconditional_inputs"] = False

        return config, input_ids, attention_mask, inputs_dict

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, filtered_inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)

        # Make sure we only return `input_ids`.
        # Note that audio_codes will still be generated internally, so the ability to test audio codes is still there.
        # There are further tests to test that audio waveforms and codes are well generated.
        filtered_inputs_dict["return_audio_waveforms"] = False
        filtered_inputs_dict["return_audio_codes"] = False
        filtered_inputs_dict["concat_unconditional_inputs"] = False

        return config, filtered_inputs_dict

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        # Overwrite because the generate method actually alway uses `inputs_embeds` so `use_cache` is always `True`
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length if idx == 0 else 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def _check_outputs(self, output, config, use_cache=False, num_return_sequences=1, num_beams=1):
        # Overwrite because the generate method actually alway uses `inputs_embeds` so `use_cache` is always `True`
        super()._check_outputs(
            output, config, use_cache=True, num_return_sequences=num_return_sequences, num_beams=num_beams
        )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        # Overwrite because the generate method actually alway uses `inputs_embeds` so `use_cache` is always `True`
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        # Overwrite because the generate method actually alway uses `inputs_embeds` so `use_cache` is always `True`
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = 1
            src_len = min_length + idx

            expected_shape = (
                batch_size * num_beam_groups,
                config.num_attention_heads,
                tgt_len,
                src_len,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv", "input_proj", "output_proj"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @unittest.skip(reason="Continuing from past key values is not straightforward as we're dealing with 3 inputs")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Moshi doesn't support contrastive generation yet.")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Moshi doesn't support contrastive generation yet.")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Moshi doesn't support contrastive generation yet.")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Adapting this test is costly. `test_eager_matches_sdpa_generate` tests this already.")
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        pass

    @unittest.skip(reason="The Moshi model does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # Then, test left-padding

        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, input_dict = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            next_logits_wo_padding = model(input_ids=input_ids, attention_mask=attention_mask, **input_dict).logits[
                :, -1, :
            ]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)

            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)

            padding = (
                torch.ones(
                    (pad_size[0], self.model_tester.num_codebooks, 32), dtype=input_ids.dtype, device=torch_device
                )
                * config.audio_vocab_size
            )
            padded_moshi_audio_codes = torch.cat((padding, input_dict["moshi_audio_codes"]), dim=2)
            padded_user_audio_codes = torch.cat((padding, input_dict["user_audio_codes"]), dim=2)

            model_kwargs = {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "moshi_audio_codes": padded_moshi_audio_codes,
                "user_audio_codes": padded_user_audio_codes,
            }

            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)

    @require_torch_sdpa
    @slow
    @is_flaky(max_attempts=5, description="flaky on some models.")
    def test_eager_matches_sdpa_generate(self):
        """Overwritten -- mochi has custom inputs and custom output checks"""

        max_new_tokens = 5

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            inputs_dict[model_class.main_input_name] = dummy_input

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model_sdpa = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                ).to(torch_device)

                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        has_sdpa = True
                        break
                if not has_sdpa:
                    raise ValueError("The SDPA model should have SDPA attention layers")

                # Just test that a large cache works as expected
                res_eager = model_eager.generate(
                    **inputs_dict,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    depth_decoder_do_sample=False,
                )

                res_sdpa = model_sdpa.generate(
                    **inputs_dict,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    depth_decoder_do_sample=False,
                )

                torch.testing.assert_close(res_eager.sequences, res_sdpa.sequences)
                torch.testing.assert_close(res_eager.audio_sequences, res_sdpa.audio_sequences)

    @pytest.mark.generate
    def test_generate_without_input_ids(self):
        config, _, _, _ = self._get_input_ids_and_config()

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(
                do_sample=False, max_new_tokens=self.max_new_tokens, remove_invalid_values=True
            )
            self.assertIsNotNone(output_ids_generate)

    @unittest.skip(reason="The audio encoder has no gradients.")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="The audio encoder has no gradients.")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="The audio encoder has no gradients.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_generate_from_input_values(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, _, _ = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()

            input_values_length = int(
                self.model_tester.seq_length * config.sampling_rate / config.audio_encoder_config.frame_rate
            )

            user_input_values = floats_tensor((input_ids.shape[0], 1, input_values_length))
            moshi_input_values = floats_tensor((input_ids.shape[0], 1, input_values_length))

            user_audio_codes = model.audio_encoder.encode(user_input_values, num_quantizers=model.num_codebooks)[0]
            moshi_audio_codes = model.audio_encoder.encode(moshi_input_values, num_quantizers=model.num_codebooks)[0]

            outputs_from_audio_codes = model.generate(
                input_ids, max_new_tokens=5, user_audio_codes=user_audio_codes, moshi_audio_codes=moshi_audio_codes
            )

            outputs_from_audio_values = model.generate(
                input_ids, max_new_tokens=5, user_input_values=user_input_values, moshi_input_values=moshi_input_values
            )

            self.assertTrue((outputs_from_audio_values.sequences == outputs_from_audio_codes.sequences).all())
            self.assertTrue(
                torch.allclose(outputs_from_audio_codes.audio_sequences, outputs_from_audio_values.audio_sequences)
            )

    def test_generate_depth_decoder_kwargs(self):
        # test sampling and beam search
        for model_class in self.all_generative_model_classes:
            config, input_ids, _, input_dict = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()

            model.generate(input_ids, max_new_tokens=5, **input_dict, depth_decoder_do_sample=True)

            model.generate(
                input_ids, max_new_tokens=5, **input_dict, depth_decoder_do_sample=True, depth_decoder_num_beams=5
            )

    def test_generate_from_unconditional(self):
        # test sampling and beam search
        for model_class in self.all_generative_model_classes:
            config, input_ids, _, input_dict = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()

            # check bs>1
            model.generate(
                **model.get_unconditional_inputs(num_samples=4), max_new_tokens=5, concat_unconditional_inputs=False
            )

            # check same results from uncondtional or no inputs
            outputs_from_unconditional = model.generate(
                **model.get_unconditional_inputs(num_samples=1), max_new_tokens=5, concat_unconditional_inputs=False
            )
            outputs_from_none = model.generate(max_new_tokens=5)

            self.assertTrue((outputs_from_unconditional.sequences == outputs_from_none.sequences).all())
            self.assertTrue(
                torch.allclose(outputs_from_unconditional.audio_sequences, outputs_from_none.audio_sequences)
            )

    @unittest.skip(reason="Compile not yet supported because in Moshi models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @is_flaky(max_attempts=5, description="flaky on some models.")
    def test_save_load(self):
        super().test_save_load()


def place_dict_on_device(dict_to_place, device):
    for key in dict_to_place:
        if dict_to_place[key] is not None and isinstance(dict_to_place[key], torch.Tensor):
            dict_to_place[key] = dict_to_place[key].to(device)
    return dict_to_place


@require_torch
class MoshiIntegrationTests(unittest.TestCase):
    @cached_property
    def feature_extractor(self):
        return AutoFeatureExtractor.from_pretrained("kmhf/hf-moshiko")

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("kmhf/hf-moshiko")

    def _load_datasample(self):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        dataset = ds.cast_column("audio", Audio(sampling_rate=self.feature_extractor.sampling_rate))
        # automatic decoding with librispeech
        speech_sample = dataset.sort("id")[0]["audio"]["array"]
        return speech_sample

    @slow
    def test_moshika_conditional_greedy(self):
        model = MoshiForConditionalGeneration.from_pretrained(
            "kmhf/hf-moshika", torch_dtype=torch.float16, device_map="auto"
        )
        inputs = self.feature_extractor(self._load_datasample(), return_tensors="pt").to(
            device=torch_device, dtype=torch.float16
        )

        user_audio_codes = model.audio_encoder.encode(**inputs, num_quantizers=8).audio_codes

        input_ids = self.tokenizer.encode("<pad><pad><pad><pad><unk> Hello,<pad><unk>", return_tensors="pt").to(
            torch_device
        )

        # fmt: off
        moshi_audio_codes = [[[1049, 127, 1880, 972, 972, 1156, 1913, 415, 1933],
                              [1700, 243, 91, 91, 91, 745, 1478, 638, 57],
                              [1626, 457, 457, 457, 457, 1839, 200, 2011, 1142],
                              [546, 290, 390, 390, 290, 1408, 1812, 1187, 1911],
                              [306, 306, 1314, 1314, 1314, 759, 796, 854, 1466],
                              [1443, 1443, 1030, 317, 347, 1178, 613, 1576, 2023],
                              [1871, 428, 1433, 1433, 1978, 1405, 1755, 820, 610],
                              [2008, 1744, 1511, 568, 1533, 550, 237, 1412, 1401]]]
        # fmt: on

        moshi_audio_codes = torch.tensor(moshi_audio_codes, device=torch_device)
        user_audio_codes = user_audio_codes[:, :, : moshi_audio_codes.shape[-1]]

        model_outputs = model.generate(
            user_audio_codes=user_audio_codes,
            moshi_audio_codes=moshi_audio_codes,
            input_ids=input_ids,
            do_sample=False,
            depth_decoder_do_sample=False,
            return_audio_codes=True,
            max_new_tokens=2,
        )

        expected_text_token = 452
        expected_audio_tokens = [916, 1396, 1238, 579, 1105, 914, 1257, 810]  # fmt: skip

        self.assertTrue(expected_text_token == model_outputs.sequences[0, -2].cpu().item())
        self.assertTrue(expected_audio_tokens == model_outputs.audio_codes[0, :, -1].cpu().tolist())

    @slow
    def test_moshiko_greedy_unconditional_fp16_eager(self):
        model = MoshiForConditionalGeneration.from_pretrained(
            "kmhf/hf-moshiko", torch_dtype=torch.float16, device_map="auto"
        )
        some_expected_audio_tokens = [[1049, 127], [1700, 243], [1626, 457], [546, 290], [306, 306], [1443, 1443], [1871, 428], [2008, 1744]]  # fmt: skip

        model_outputs = model.generate(
            do_sample=False, depth_decoder_do_sample=False, return_audio_codes=True, max_new_tokens=10
        )

        # eager equivalence is not as strict as sdpa.
        self.assertTrue(some_expected_audio_tokens == model_outputs.audio_codes[0, :, :2].cpu().tolist())

    @slow
    def test_moshiko_greedy_unconditional_fp32(self):
        model = MoshiForConditionalGeneration.from_pretrained(
            "kmhf/hf-moshiko", torch_dtype=torch.float32, device_map="auto"
        )

        expected_audio_codesum = 72065
        expected_text_tokens = [3, 3, 3, 0, 11725, 261, 3, 3, 3, 3]  # fmt: skip
        some_expected_audio_tokens = [[1049, 127], [1700, 243], [1626, 457], [546, 290], [306, 306], [1443, 1443], [1871, 428], [2008, 1744]]  # fmt: skip

        model_outputs = model.generate(
            do_sample=False, depth_decoder_do_sample=False, return_audio_codes=True, max_new_tokens=10
        )

        # make sure audio encoded codes are correct
        audio_code_sums = model_outputs.audio_codes.sum().item()
        self.assertTrue(np.abs(audio_code_sums - expected_audio_codesum) <= (3e-3 * audio_code_sums))

        self.assertTrue(expected_text_tokens == model_outputs.sequences[0, 1:].cpu().tolist())
        self.assertTrue(some_expected_audio_tokens == model_outputs.audio_codes[0, :, :2].cpu().tolist())

    @slow
    @require_torch_fp16
    def test_moshiko_greedy_unconditional_fp16(self):
        model = MoshiForConditionalGeneration.from_pretrained(
            "kmhf/hf-moshiko", torch_dtype=torch.float16, device_map="auto"
        )

        expected_audio_codesum = 72065
        expected_text_tokens = [3, 3, 3, 0, 11725, 261, 3, 3, 3, 3]  # fmt: skip
        some_expected_audio_tokens = [[1049, 127], [1700, 243], [1626, 457], [546, 290], [306, 306], [1443, 1443], [1871, 428], [2008, 1744]]  # fmt: skip

        model_outputs = model.generate(
            do_sample=False, depth_decoder_do_sample=False, return_audio_codes=True, max_new_tokens=10
        )

        # make sure audio encoded codes are correct
        audio_code_sums = model_outputs.audio_codes.sum().item()
        self.assertTrue(np.abs(audio_code_sums - expected_audio_codesum) <= (3e-3 * audio_code_sums))

        self.assertTrue(expected_text_tokens == model_outputs.sequences[0, 1:].cpu().tolist())
        self.assertTrue(some_expected_audio_tokens == model_outputs.audio_codes[0, :, :2].cpu().tolist())

    @slow
    @require_torch_fp16
    def test_moshika_greedy_unconditional_fp16(self):
        model = MoshiForConditionalGeneration.from_pretrained(
            "kmhf/hf-moshika", torch_dtype=torch.float16, device_map="auto"
        )

        expected_audio_codesum = 72932
        expected_text_tokens = [3, 3, 3, 0, 667, 263, 3, 3, 0, 705]  # fmt: skip
        some_expected_audio_tokens = [[1049, 127], [1700, 243], [1626, 457], [546, 290], [306, 306], [1443, 347], [1871, 428], [2008, 2008]]  # fmt: skip

        model_outputs = model.generate(
            do_sample=False, depth_decoder_do_sample=False, return_audio_codes=True, max_new_tokens=10
        )

        # make sure audio encoded codes are correct
        audio_code_sums = model_outputs.audio_codes.sum().item()
        self.assertTrue(np.abs(audio_code_sums - expected_audio_codesum) <= 2048)

        self.assertTrue(expected_text_tokens == model_outputs.sequences[0, 1:].cpu().tolist())
        self.assertTrue(some_expected_audio_tokens == model_outputs.audio_codes[0, :, :2].cpu().tolist())
