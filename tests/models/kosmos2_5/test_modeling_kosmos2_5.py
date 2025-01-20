# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch KOSMOS-2.5 model."""

import copy
import inspect
import os
import tempfile
import unittest

import numpy as np
import pytest
import requests
from parameterized import parameterized

from transformers import AutoProcessor, Kosmos2_5Config
from transformers.models.kosmos2_5.configuration_kosmos2_5 import (
    Kosmos2_5TextConfig,
    Kosmos2_5VisionConfig,
)
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Kosmos2_5ForConditionalGeneration, Kosmos2_5Model


if is_vision_available():
    from PIL import Image


class Kosmos2_5VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=6,
        image_size=32,
        patch_size=4,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        d_ff=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0,
        attention_dropout=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.d_ff = d_ff
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_embed_hidden_size = patch_size * patch_size * num_channels
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        flattened_patches = floats_tensor([self.batch_size, self.seq_length, self.patch_embed_hidden_size + 2])
        config = self.get_config()

        return config, flattened_patches

    def get_config(self):
        return Kosmos2_5VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            d_ff=self.d_ff,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            patch_embed_hidden_size=self.patch_embed_hidden_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, flattened_patches = config_and_inputs
        inputs_dict = {"flattened_patches": flattened_patches}
        return config, inputs_dict


class Kosmos2_5TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=6,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0,
        attention_dropout=0,
        max_position_embeddings=512,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Kosmos2_5TextConfig(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            ffn_dim=self.ffn_dim,
            layers=self.num_hidden_layers,
            attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Kosmos2_5ModelTester:
    def __init__(
        self,
        parent,
        text_kwargs=None,
        vision_kwargs=None,
        latent_query_num=3,
        is_training=True,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Kosmos2_5TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Kosmos2_5VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length
        self.latent_query_num = latent_query_num
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, flattened_patches = self.vision_model_tester.prepare_config_and_inputs()

        # build `image_embeds_position_mask`
        image_embeds_position_mask = torch.zeros_like(input_ids)
        image_embeds_position_mask[:, 1 : 1 + self.latent_query_num :] = 1

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            image_embeds_position_mask,
            flattened_patches,
        )

    def get_config(self):
        return Kosmos2_5Config(
            self.text_model_tester.get_config().to_dict(),
            self.vision_model_tester.get_config().to_dict(),
            latent_query_num=self.latent_query_num,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        image_embeds_position_mask,
        flattened_patches,
    ):
        model = Kosmos2_5Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, flattened_patches, image_embeds_position_mask, attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (
                self.text_model_tester.batch_size,
                self.text_model_tester.seq_length,
                self.text_model_tester.hidden_size,
            ),
        )
        self.parent.assertEqual(
            result.image_embeds.shape,
            (
                self.text_model_tester.batch_size,
                self.latent_query_num,
                self.text_model_tester.hidden_size,
            ),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            image_embeds_position_mask,
            flattened_patches,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_embeds_position_mask": image_embeds_position_mask,
            "flattened_patches": flattened_patches,
        }
        return config, inputs_dict


@require_torch
class Kosmos2_5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Kosmos2_5Model, Kosmos2_5ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Kosmos2_5ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Kosmos2_5Model,
            "image-to-text": Kosmos2_5ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    # TODO: `image-to-text` pipeline for this model needs Processor.
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_casse_name,
        config_class,
        model_architecture,
        tokenizer_name,
        processor_name,
    ):
        return pipeline_test_casse_name == "ImageToTextPipelineTests"

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class.__name__ == "Kosmos2_5ForConditionalGeneration":
                inputs_dict["labels"] = torch.zeros(
                    (
                        self.model_tester.text_model_tester.batch_size,
                        self.model_tester.text_model_tester.seq_length,
                    ),
                    dtype=torch.long,
                    device=torch_device,
                )

        if model_class.__name__ in [
            "Kosmos2_5Model",
            "Kosmos2_5ForConditionalGeneration",
        ]:
            bs, _ = inputs_dict["input_ids"].shape
            seqlen = self.model_tester.text_model_tester.seq_length
            inputs_dict["input_ids"] = torch.arange(seqlen, device=torch_device).unsqueeze(0).expand(bs, seqlen)
            inputs_dict["input_ids"] = inputs_dict["input_ids"] % self.model_tester.text_model_tester.vocab_size
            inputs_dict["attention_mask"] = torch.ones((bs, seqlen), device=torch_device)
            inputs_dict["image_embeds_position_mask"] = torch.zeros((bs, seqlen), device=torch_device)
            inputs_dict["image_embeds_position_mask"][:, : self.model_tester.latent_query_num] = 1
        return inputs_dict

    def setUp(self):
        self.model_tester = Kosmos2_5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Kosmos2_5Config, hidden_size=37)

    # overwrite from common to skip `image_to_text_projection.latent_query`
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == "image_to_text_projection.latent_query":
                        # The original code use ` nn.Parameter(torch.randn(...))` for which this test won't pass.
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_save_without_tied_weights(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.tie_word_embeddings = False
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v,
                        reloaded_state[k],
                        msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}",
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

    # overwrite from common in order to use `self.model_tester.text_model_tester.num_hidden_layers`
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester,
                "expected_num_hidden_layers",
                self.model_tester.text_model_tester.num_hidden_layers + 1,
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.text_model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.text_model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # overwrite from common in order to use `config.text_config.vocab_size` instead of `config.vocab_size`
    def test_tie_model_weights(self):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # embeddings.weight.data.div_(2)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # decoding.weight.data.div_(4)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.text_config.vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

            # decoding.weight.data.mul_(20)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
            # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))

    @slow
    def test_model_from_pretrained(self):
        model_name = "microsoft/kosmos-2.5"
        model = Kosmos2_5Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        super().test_model_parallelism()

    # TODO: ydshieh
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    @unittest.skip(reason="kosmos-2.5 flash attention does not support right padding")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    # TODO: ydshieh
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    @unittest.skip(reason="kosmos-2.5 test : the dummy inputs should be tweaked: dummy_input = inputs_dict")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    # TODO: ydshieh
    @require_torch_sdpa
    @require_torch_gpu
    @slow
    @unittest.skip(reason="_update_causal_mask is not implemented yet which fails this test")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    # TODO: ydshieh
    @unittest.skip(reason="doesn't support padding yet")
    def test_eager_matches_sdpa_inference_1_bfloat16(self):
        pass

    # TODO: ydshieh
    @unittest.skip(reason=" the model hasn't been added to auto class")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("This test is currently not well designed for multimodal model (float type as an input).")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("This test is currently not well designed for multimodal model (float type as an input).")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Kosmos 2.5 is multimodel and has specific input shapes.")
    def test_flash_attn_2_generate_reuse_cache(self):
        pass

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip(
        "KOSMOS-2.5 doesn't support inputs embeds. The test isn't skipped by checking input args because KOSMOS-2 has `generate()` overwritten",
    )
    def test_generate_from_inputs_embeds(self):
        pass

    # TODO: ydshieh
    @pytest.mark.generate
    @unittest.skip(
        "Kosmos2_5ForConditionalGeneration returns `vision_model_output` which is currently not working with `stack_model_outputs`",
    )
    def test_beam_search_low_memory(self):
        pass

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # Overwrite because Kosmos-2.5 need to padd pixel values and pad image-attn-mask

        def _prepare_model_kwargs(input_ids, attention_mask, pad_size, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            if "image_embeds_position_mask" in signature:
                image_embeds_position_mask = torch.zeros_like(input_ids)
                image_embeds_position_mask[:, (pad_size + 1) : pad_size + 1 + self.model_tester.latent_query_num] = 1
                model_kwargs["image_embeds_position_mask"] = image_embeds_position_mask
            return model_kwargs

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]
            flattened_patches = inputs_dict["flattened_patches"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, pad_size=0, signature=signature)
            next_logits_wo_padding = model(**model_kwargs, flattened_patches=flattened_patches).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(
                padded_input_ids, padded_attention_mask, pad_size=32, signature=signature
            )
            next_logits_with_padding = model(**model_kwargs, flattened_patches=flattened_patches).logits[:, -1, :]

            # They should result in very similar logits
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=1e-3))

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(
                    main_input,
                    inputs["flattened_patches"],
                    inputs["image_embeds_position_mask"],
                )
                traced_model = torch.jit.trace(
                    model,
                    (
                        main_input,
                        inputs["flattened_patches"],
                        inputs["image_embeds_position_mask"],
                    ),
                )
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


@require_vision
@require_torch
@slow
class Kosmos2_5ModelIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    def run_example(self, prompt, image, model, processor):
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        _, _ = inputs.pop("height"), inputs.pop("width")
        inputs = {k: v.to(torch_device) if v is not None else None for k, v in inputs.items()}
        inputs["flattened_patches"] = inputs["flattened_patches"].to(model.dtype)

        generation_outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
        )
        generated_ids = generation_outputs
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_ids, generated_text

    def test_eager(self):
        url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        image = Image.open(requests.get(url, stream=True).raw)

        dtype = torch.bfloat16
        repo = "microsoft/kosmos-2.5"
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo, device_map=torch_device, torch_dtype=dtype, attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(repo)
        prompt = "<ocr>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        EXPECTED_TEXT = {
            7: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_651></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_642></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_683></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n"
            ],
            8: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_650></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_644></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_687></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

        prompt = "<md>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)

        EXPECTED_TEXT = {
            7: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
            8: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

    def test_sdpa(self):
        url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        image = Image.open(requests.get(url, stream=True).raw)

        dtype = torch.bfloat16
        repo = "microsoft/kosmos-2.5"
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo, device_map=torch_device, torch_dtype=dtype, attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(repo)
        prompt = "<ocr>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        EXPECTED_TEXT = {
            7: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_651></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_642></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_683></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n",
            ],
            8: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_651></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_642></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_683></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

        prompt = "<md>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)

        EXPECTED_TEXT = {
            7: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
            8: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_FA2(self):
        url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        image = Image.open(requests.get(url, stream=True).raw)

        dtype = torch.bfloat16
        repo = "microsoft/kosmos-2.5"
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo,
            device_map=torch_device,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(repo)
        prompt = "<ocr>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        EXPECTED_TEXT = [
            "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_612></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_812><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_650></bbox>1\n<bbox><x_79><y_614><x_468><y_650></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_610><x_813><y_644></bbox>0\n<bbox><x_50><y_658><x_65><y_693></bbox>1\n<bbox><x_76><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_815><y_687></bbox>0\n<bbox><x_31><y_742><x_822><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_780><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_874></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_835><y_1108></bbox>Card Payment 50,000\n"
        ]

        self.assertListEqual(generated_text, EXPECTED_TEXT)

        prompt = "<md>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        # A10 gives the 1st one, but A100 gives the 2nd one
        EXPECTED_TEXT = [
            "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n\n<table>\n<thead>\n<tr>\n<th>\nSub Total\n</th>\n<th>\n45,455\n</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td>\nPB1 (10%)\n</td>\n<td>\n4,545\n</td>\n</tr>\n<tr>\n<td>\nRounding\n</td>\n<td>\n0\n</td>\n</tr>\n<tr>\n<td>\n<strong>\nTotal\n</strong>\n</td>\n<td>\n<strong>\n50,000\n</strong>\n</td>\n</tr>\n</tbody>\n</table>\n\nCard Payment 50,000",
            "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n",
        ]
        self.assertIn(generated_text[0], EXPECTED_TEXT)
