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
"""Testing suite for the PyTorch BLIP-2 model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
import pytest
import requests
from parameterized import parameterized

from transformers import CONFIG_MAPPING, Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_torch_sdpa_available, is_vision_available

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
    from torch import nn

    from transformers import (
        Blip2ForConditionalGeneration,
        Blip2ForImageTextRetrieval,
        Blip2Model,
        Blip2TextModelWithProjection,
        Blip2VisionModel,
        Blip2VisionModelWithProjection,
    )


if is_vision_available():
    from PIL import Image

    from transformers import Blip2Processor


class Blip2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Blip2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = Blip2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Blip2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as BLIP-2's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Blip2VisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Blip2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Blip2VisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="BLIP-2's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip
    def test_training(self):
        pass

    @unittest.skip
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Blip2VisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Blip2VisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip2-opt-2.7b"
        model = Blip2VisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class Blip2QFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        bos_token_id=0,
        scope=None,
        use_qformer_text_input=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = bos_token_id
        self.use_qformer_text_input = use_qformer_text_input

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
        return Blip2QFormerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
            use_qformer_text_input=self.use_qformer_text_input,
        )


# this class is based on `OPTModelTester` found in tests/models/opt/test_modeling_opt.py
class Blip2TextModelDecoderOnlyTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        embed_dim=16,
        num_labels=3,
        word_embed_proj_dim=16,
        type_sequence_label_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.word_embed_proj_dim = word_embed_proj_dim
        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(3)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        attention_mask = input_ids.ne(self.pad_token_id)

        return config, input_ids, attention_mask

    def get_config(self):
        return CONFIG_MAPPING["opt"](
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            is_encoder_decoder=False,
            word_embed_proj_dim=self.word_embed_proj_dim,
        )


# this model tester uses a decoder-only language model (OPT)
class Blip2ForConditionalGenerationDecoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vision_kwargs=None,
        qformer_kwargs=None,
        text_kwargs=None,
        is_training=True,
        num_query_tokens=10,
        image_token_index=4,
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = Blip2TextModelDecoderOnlyTester(parent, **text_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length + num_query_tokens  # need seq_length for common tests
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens
        self.image_token_index = image_token_index

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()

        vision_tokens = (
            torch.ones((input_ids.shape[0], self.num_query_tokens), device=torch_device, dtype=input_ids.dtype)
            * self.image_token_index
        )
        input_ids[input_ids == self.image_token_index] = self.text_model_tester.pad_token_id
        input_ids = torch.cat([vision_tokens, input_ids], dim=-1)
        vision_attention_mask = torch.ones_like(vision_tokens)
        attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=-1)

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
            image_token_index=self.image_token_index,
        )

    def create_and_check_for_conditional_generation(self, config, input_ids, attention_mask, pixel_values):
        model = Blip2ForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, attention_mask)

        expected_seq_length = self.num_query_tokens + self.text_model_tester.seq_length
        self.parent.assertEqual(
            result.logits.shape,
            (self.vision_model_tester.batch_size, expected_seq_length, self.text_model_tester.vocab_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Blip2ForConditionalGenerationDecoderOnlyTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2ForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (Blip2ForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = True
    _is_composite = True

    def setUp(self):
        self.model_tester = Blip2ForConditionalGenerationDecoderOnlyModelTester(self)
        common_properties = ["image_token_index", "num_query_tokens", "image_text_hidden_size"]
        self.config_tester = ConfigTester(
            self, config_class=Blip2Config, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    def _create_and_check_torchscript(self, config, inputs_dict):
        # overwrite because BLIP requires ipnut ids and pixel values as input
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to `False`")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            for attn_implementation in ["eager", "sdpa"]:
                if attn_implementation == "sdpa" and (not model_class._supports_sdpa or not is_torch_sdpa_available()):
                    continue

                configs_no_init._attn_implementation = attn_implementation
                model = model_class(config=configs_no_init)
                model.to(torch_device)
                model.eval()
                inputs = self._prepare_for_class(inputs_dict, model_class)

                main_input_name = model_class.main_input_name

                try:
                    if model.config.is_encoder_decoder:
                        model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                        main_input = inputs[main_input_name]
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs["attention_mask"]
                        decoder_input_ids = inputs["decoder_input_ids"]
                        decoder_attention_mask = inputs["decoder_attention_mask"]
                        model(main_input, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
                        traced_model = torch.jit.trace(
                            model, (main_input, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
                        )
                    else:
                        main_input = inputs[main_input_name]
                        input_ids = inputs["input_ids"]

                        if model.config._attn_implementation == "sdpa":
                            trace_input = {main_input_name: main_input, "input_ids": input_ids}

                            if "attention_mask" in inputs:
                                trace_input["attention_mask"] = inputs["attention_mask"]
                            else:
                                self.skipTest(reason="testing SDPA without attention_mask is not supported")

                            model(main_input, attention_mask=inputs["attention_mask"])
                            # example_kwarg_inputs was introduced in torch==2.0, but it is fine here since SDPA has a requirement on torch>=2.1.
                            traced_model = torch.jit.trace(model, example_kwarg_inputs=trace_input)
                        else:
                            model(main_input, input_ids)
                            traced_model = torch.jit.trace(model, (main_input, input_ids))
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

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Blip2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_to_base(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
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

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.vision_model._supports_sdpa else "eager"
                qformer_attn = "sdpa" if model.qformer._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.vision_model.config._attn_implementation == vision_attn)
                self.assertTrue(model.qformer.config._attn_implementation == qformer_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.qformer.config._attn_implementation == "eager")

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
                if not has_sdpa and any(
                    module_attn == "sdpa" for module_attn in [text_attn, vision_attn, qformer_attn]
                ):
                    raise ValueError("The SDPA model should have SDPA attention layers")

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Blip2Config and check if we can load Blip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Blip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Blip2Config and check if we can load Blip2QFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = Blip2QFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.assertIsNotNone(model)

    # overwrite because BLIP internally calls LM.generate() with embeds thus it cannot operate in no cache format
    def _check_outputs(self, output, config, use_cache=False, num_return_sequences=1, num_beams=1):
        use_cache = True  # force this to be True in case False is passed

        input_batch_size = int(output.sequences.shape[0] / num_return_sequences)
        internal_batch_size = (
            input_batch_size * num_beams if num_beams > 1 else input_batch_size * num_return_sequences
        )

        seq_length = getattr(self.model_tester, "seq_length", None)
        seq_length = getattr(self.model_tester, "encoder_seq_length", seq_length)
        seq_length = getattr(self.model_tester, "text_seq_length", seq_length)

        config = config.text_config if hasattr(config, "text_config") else config

        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # in some models we subsample the sequence length in inner layers
        if hasattr(self.model_tester, "get_subsampled_output_lengths"):
            seq_length = self.model_tester.get_subsampled_output_lengths(seq_length)

        # scores
        self._check_scores(internal_batch_size, output.scores, length=gen_len, config=config)

        # unprocessed logits
        self._check_logits(internal_batch_size, output.logits, config=config)

        # Attentions
        if self.has_attentions:
            if config.is_encoder_decoder:
                # encoder
                self._check_encoder_attention_for_generate(
                    output.encoder_attentions, input_batch_size, config, seq_length
                )
                # decoder
                self._check_attentions_for_generate(
                    internal_batch_size,
                    output.decoder_attentions,
                    min_length=1,
                    max_length=output.sequences.shape[-1],
                    config=config,
                    use_cache=use_cache,
                )
            else:
                # if use_cache first input is equal to no use_cache, so skip here
                attentions = output.attentions if not use_cache else output.attentions[1:]
                min_length = seq_length if not use_cache else seq_length + 1
                self._check_attentions_for_generate(
                    internal_batch_size,
                    attentions=attentions,
                    min_length=min_length,
                    max_length=output.sequences.shape[-1],
                    config=config,
                    use_cache=use_cache,
                )

        # Hidden States
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_hidden_states_for_generate(
                output.encoder_hidden_states, input_batch_size, config, seq_length
            )

            # decoder
            self._check_hidden_states_for_generate(
                internal_batch_size,
                output.decoder_hidden_states,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            hidden_states = output.hidden_states if not use_cache else output.hidden_states[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_hidden_states_for_generate(
                internal_batch_size,
                hidden_states,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Past Key Value States
        if use_cache:
            past_key_values = output.past_key_values
            past_sequence_length = output.sequences.shape[-1] - 1
            self._check_past_key_values_for_generate(
                internal_batch_size,
                past_key_values,
                seq_length=past_sequence_length,
                config=config,
            )

    # overwrite because BLIP2 cannot generate only from input ids, and requires pixel values in all cases to be present
    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict().keys()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict.get("attention_mask")
            pixel_values = inputs_dict["pixel_values"]
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs, pixel_values=pixel_values).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs, pixel_values=pixel_values).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)

    @unittest.skip("BLIP2 cannot generate only from input ids, and requires pixel values in all cases to be present")
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass


# this class is based on `T5ModelTester` found in tests/models/t5/test_modeling_t5.py
class Blip2TextModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=12,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_config(self):
        return CONFIG_MAPPING["t5"](
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )


# this model tester uses an encoder-decoder language model (T5)
class Blip2ModelTester:
    def __init__(
        self, parent, vision_kwargs=None, qformer_kwargs=None, text_kwargs=None, is_training=True, num_query_tokens=10
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = Blip2TextModelTester(parent, **text_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length  # need seq_length for common tests
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        (
            _,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, lm_labels

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, labels
    ):
        model = Blip2ForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

        self.parent.assertEqual(
            result.logits.shape,
            (
                self.vision_model_tester.batch_size,
                self.text_model_tester.seq_length,
                self.text_model_tester.vocab_size,
            ),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


@require_torch
class Blip2ModelTest(ModelTesterMixin, PipelineTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2ForConditionalGeneration, Blip2Model) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Blip2Model,
            "image-to-text": Blip2ForConditionalGeneration,
            "visual-question-answering": Blip2ForConditionalGeneration,
            "image-text-to-text": Blip2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = True
    test_attention_outputs = False
    test_torchscript = True
    _is_composite = True

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if pipeline_test_case_name == "VisualQuestionAnsweringPipelineTests":
            # Get `RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'`.
            return True

        return False

    def setUp(self):
        self.model_tester = Blip2ModelTester(self)
        common_properties = ["image_token_index", "num_query_tokens", "image_text_hidden_size"]
        self.config_tester = ConfigTester(
            self, config_class=Blip2Config, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    def _create_and_check_torchscript(self, config, inputs_dict):
        # overwrite because BLIP requires ipnut ids and pixel values as input
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to `False`")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            for attn_implementation in ["eager", "sdpa"]:
                if attn_implementation == "sdpa" and (not model_class._supports_sdpa or not is_torch_sdpa_available()):
                    continue

                configs_no_init._attn_implementation = attn_implementation
                model = model_class(config=configs_no_init)
                model.to(torch_device)
                model.eval()
                inputs = self._prepare_for_class(inputs_dict, model_class)

                main_input_name = model_class.main_input_name

                try:
                    if model.config.is_encoder_decoder:
                        model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                        main_input = inputs[main_input_name]
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs["attention_mask"]
                        decoder_input_ids = inputs["decoder_input_ids"]
                        decoder_attention_mask = inputs["decoder_attention_mask"]
                        model(main_input, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
                        traced_model = torch.jit.trace(
                            model, (main_input, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
                        )
                    else:
                        main_input = inputs[main_input_name]
                        input_ids = inputs["input_ids"]

                        if model.config._attn_implementation == "sdpa":
                            trace_input = {main_input_name: main_input, "input_ids": input_ids}

                            if "attention_mask" in inputs:
                                trace_input["attention_mask"] = inputs["attention_mask"]
                            else:
                                self.skipTest(reason="testing SDPA without attention_mask is not supported")

                            model(main_input, attention_mask=inputs["attention_mask"])
                            # example_kwarg_inputs was introduced in torch==2.0, but it is fine here since SDPA has a requirement on torch>=2.1.
                            traced_model = torch.jit.trace(model, example_kwarg_inputs=trace_input)
                        else:
                            model(main_input, input_ids)
                            traced_model = torch.jit.trace(model, (main_input, input_ids))
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

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Blip2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base Blip2Model")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
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

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.vision_model._supports_sdpa else "eager"
                qformer_attn = "sdpa" if model.qformer._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.vision_model.config._attn_implementation == vision_attn)
                self.assertTrue(model.qformer.config._attn_implementation == qformer_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.qformer.config._attn_implementation == "eager")

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
                if not has_sdpa and any(
                    module_attn == "sdpa" for module_attn in [text_attn, vision_attn, qformer_attn]
                ):
                    raise ValueError("The SDPA model should have SDPA attention layers")

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Blip2Config and check if we can load Blip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Blip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Blip2Config and check if we can load Blip2QFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = Blip2QFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_get_text_features(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict = {
            "input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
            "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(torch_device),
            "decoder_input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
        }

        model = Blip2Model(config).to(torch_device)
        model.eval()
        text_features = model.get_text_features(**inputs_dict)
        self.assertEqual(text_features[0].shape, (1, 10, config.text_config.vocab_size))

    def test_get_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = Blip2Model(config).to(torch_device)
        model.eval()
        image_features = model.get_image_features(**inputs_dict)
        self.assertEqual(
            image_features[0].shape,
            (
                self.model_tester.vision_model_tester.batch_size,
                self.model_tester.vision_model_tester.seq_length,
                config.vision_config.hidden_size,
            ),
        )

    def test_get_qformer_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = Blip2Model(config).to(torch_device)
        model.eval()
        qformer_features = model.get_qformer_features(**inputs_dict)
        self.assertEqual(
            qformer_features[0].shape,
            (self.model_tester.vision_model_tester.batch_size, 10, config.vision_config.hidden_size),
        )

    # override from common to deal with nested configurations (`vision_config`, `text_config` and `qformer_config`)
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for key in ["vision_config", "qformer_config", "text_config"]:
            setattr(configs_no_init, key, _config_zero_init(getattr(configs_no_init, key)))
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


class Blip2TextModelWithProjectionTester:
    def __init__(self, parent, vision_kwargs=None, qformer_kwargs=None, is_training=True):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {"use_qformer_text_input": True}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.is_training = is_training
        self.batch_size = self.vision_model_tester.batch_size  # need bs for batching_equivalence test

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
        )

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.qformer_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Blip2TextModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.vision_model_tester.batch_size, input_ids.shape[1], self.qformer_model_tester.hidden_size),
        )
        self.parent.assertEqual(
            result.text_embeds.shape,
            (
                self.vision_model_tester.batch_size,
                input_ids.shape[1],
                config.image_text_hidden_size,
            ),
        )

        with torch.no_grad():
            result2 = model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=not config.use_return_dict,
                output_attentions=True,
                output_hidden_states=True,
            )

        self.parent.assertTrue(torch.allclose(result.text_embeds, result2[0]))
        self.parent.assertTrue(torch.allclose(result.last_hidden_state, result2[1]))
        self.parent.assertTrue(torch.allclose(result.hidden_states[0], result2[2][0]))
        self.parent.assertTrue(torch.allclose(result.hidden_states[1], result2[2][1]))
        self.parent.assertTrue(torch.allclose(result.attentions[0], result2[3][0]))
        self.parent.assertTrue(torch.allclose(result.attentions[1], result2[3][1]))


@require_torch
class Blip2TextModelWithProjectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2TextModelWithProjection,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    test_resize_embeddings = True
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Blip2TextModelWithProjectionTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Training is not yet supported")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Blip2TextModelWithProjection does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Blip2TextModelWithProjection does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Blip2TextModelWithProjection does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Blip2TextModelWithProjection has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Blip2TextModelWithProjection has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids", "attention_mask", "position_ids"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @slow
    @require_torch_accelerator
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        model = Blip2TextModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "text_projection"))

        _, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()

        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(
            outputs.text_embeds.shape,
            (
                self.model_tester.qformer_model_tester.batch_size,
                input_ids.shape[1],
                model.config.image_text_hidden_size,
            ),
        )


class Blip2VisionModelWithProjectionTester:
    def __init__(self, parent, vision_kwargs=None, qformer_kwargs=None, is_training=True):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {"use_qformer_text_input": True}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.is_training = is_training
        self.num_hidden_layers = self.vision_model_tester.num_hidden_layers
        self.num_attention_heads = self.vision_model_tester.num_attention_heads
        self.seq_length = self.vision_model_tester.seq_length
        self.hidden_size = self.vision_model_tester.hidden_size
        self.batch_size = self.vision_model_tester.batch_size  # need bs for batching_equivalence test

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
        )

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values):
        model = Blip2VisionModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values, output_attentions=True, output_hidden_states=True)

        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (
                self.vision_model_tester.batch_size,
                self.vision_model_tester.seq_length,
                self.qformer_model_tester.hidden_size,
            ),
        )
        self.parent.assertEqual(
            result.image_embeds.shape,
            (
                self.vision_model_tester.batch_size,
                config.vision_config.hidden_size,
                config.image_text_hidden_size,
            ),
        )

        with torch.no_grad():
            result2 = model(
                pixel_values,
                return_dict=not config.use_return_dict,
                output_attentions=True,
                output_hidden_states=True,
            )

        self.parent.assertTrue(torch.allclose(result.image_embeds, result2[0]))
        self.parent.assertTrue(torch.allclose(result.last_hidden_state, result2[1]))
        self.parent.assertTrue(torch.allclose(result.hidden_states[0], result2[2][0]))
        self.parent.assertTrue(torch.allclose(result.hidden_states[1], result2[2][1]))
        self.parent.assertTrue(torch.allclose(result.attentions[0], result2[3][0]))
        self.parent.assertTrue(torch.allclose(result.attentions[1], result2[3][1]))


@require_torch
class Blip2VisionModelWithProjectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2VisionModelWithProjection,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Blip2VisionModelWithProjectionTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Training is not yet supported")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Blip2VisionModelWithProjection does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Blip2VisionModelWithProjection does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @unittest.skip(reason="Blip2VisionModelWithProjection has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Blip2VisionModelWithProjection has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @slow
    @require_torch_gpu
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        model = Blip2VisionModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "vision_projection"))

        _, pixel_values = self.model_tester.prepare_config_and_inputs()

        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        self.assertEqual(
            outputs.image_embeds.shape,
            (
                self.model_tester.vision_model_tester.batch_size,
                model.config.num_query_tokens,
                model.config.image_text_hidden_size,
            ),
        )


class Blip2TextRetrievalModelTester:
    def __init__(self, parent, vision_kwargs=None, qformer_kwargs=None, is_training=True):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {"use_qformer_text_input": True}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = Blip2QFormerModelTester(parent, **qformer_kwargs)
        self.is_training = is_training
        self.batch_size = self.vision_model_tester.batch_size  # need bs for batching_equivalence test

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
        )

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.qformer_model_tester.prepare_config_and_inputs()
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = Blip2ForImageTextRetrieval(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, attention_mask, use_image_text_matching_head=True)

        self.parent.assertEqual(
            result.logits_per_image.shape,
            (self.vision_model_tester.batch_size, 2),
        )

        with torch.no_grad():
            result = model(pixel_values, input_ids, attention_mask)

        self.parent.assertEqual(
            result.logits_per_image.shape,
            (self.vision_model_tester.batch_size, self.qformer_model_tester.batch_size),
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.qformer_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


@require_torch
class Blip2TextRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2ForImageTextRetrieval,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = True
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Blip2TextRetrievalModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Blip2ForImageTextRetrieval does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Blip2Model does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values", "input_ids", "attention_mask"]
            expected_arg_names.extend(
                ["use_image_text_matching_head"] if "use_image_text_matching_head" in arg_names else []
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Blip2Config and check if we can load Blip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Blip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Blip2Config and check if we can load Blip2QFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = Blip2QFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    @require_torch_gpu
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        self.assertIsNotNone(model)

        _, input_ids, attention_mask, pixel_values = self.model_tester.prepare_config_and_inputs()

        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_image_text_matching_head=True,
            )
        self.assertEqual(outputs.logits_per_image.shape, (self.model_tester.qformer_model_tester.batch_size, 2))

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        self.assertEqual(
            outputs.logits_per_image.shape,
            (self.model_tester.vision_model_tester.batch_size, self.model_tester.qformer_model_tester.batch_size),
        )

    @unittest.skip(reason="Training is not yet supported")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Training is not yet supported")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif name == "temp":
                        self.assertAlmostEqual(
                            param.data.item(),
                            0.07,
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_torch
@slow
class Blip2ModelIntegrationTest(unittest.TestCase):
    def test_inference_opt(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(torch_device)

        # prepare image
        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        expected_ids = [50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 2, 102, 693, 2828, 15, 5, 4105, 19, 10, 2335, 50118]  # fmt: skip
        self.assertEqual(predictions[0].tolist(), expected_ids)
        self.assertEqual("a woman sitting on the beach with a dog", generated_text)

        # image and context
        prompt = "Question: which city is this? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)

        # max_length for BLIP includes prompt length from now on, use max_new_tokens
        predictions = model.generate(**inputs, max_new_tokens=11)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        expected_ids = [50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 2, 45641, 35, 61, 343, 16, 42, 116, 31652, 35, 24, 18, 45, 10, 343, 6, 24, 18, 10, 4105, 50118]  # fmt: skip
        self.assertEqual(predictions[0].tolist(), expected_ids)
        self.assertEqual(generated_text, "Question: which city is this? Answer: it's not a city, it's a beach")

    def test_inference_interpolate_pos_encoding(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(torch_device)
        processor.image_processor.size = {"height": 500, "width": 500}

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs, interpolate_pos_encoding=True)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        expected_ids = [50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 2, 102, 693, 8, 2335, 15, 5, 4105, 50118]  # fmt: skip
        self.assertEqual(predictions[0].tolist(), expected_ids)
        self.assertEqual(generated_text, "a woman and dog on the beach")

    def test_inference_opt_batched_beam_search(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(torch_device)

        # prepare image
        image = prepare_img()
        inputs = processor(images=[image, image], return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs, num_beams=2)

        # Test output (in this case, slightly different from greedy search)
        expected_ids = [50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 2, 102, 693, 2828, 15, 5, 4105, 19, 69, 2335, 50118]  # fmt: skip
        self.assertEqual(predictions[0].tolist(), expected_ids)
        self.assertEqual(predictions[1].tolist(), expected_ids)

    def test_inference_t5(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
        ).to(torch_device)

        # prepare image
        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(predictions[0].tolist(), [0, 2335, 1556, 28, 1782, 30, 8, 2608, 1])
        self.assertEqual("woman playing with dog on the beach", generated_text)

        # image and context
        prompt = "Question: which city is this? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(predictions[0].tolist(), [0, 3, 7, 152, 67, 839, 1])
        self.assertEqual(generated_text, "san diego")

    def test_inference_t5_batched_beam_search(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
        ).to(torch_device)

        # prepare image
        image = prepare_img()
        inputs = processor(images=[image, image], return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs, num_beams=2)

        # Test output (in this case, slightly different from greedy search)
        self.assertEqual(predictions[0].tolist(), [0, 2335, 1556, 28, 1782, 30, 8, 2608, 1])
        self.assertEqual(predictions[1].tolist(), [0, 2335, 1556, 28, 1782, 30, 8, 2608, 1])

    @require_torch_multi_accelerator
    def test_inference_opt_multi_accelerator(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="balanced"
        )

        # prepare image
        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(0, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(predictions[0].tolist(), [2, 102, 693, 2828, 15, 5, 4105, 19, 10, 2335, 50118])
        self.assertEqual("a woman sitting on the beach with a dog", generated_text)

        # image and context
        prompt = "Question: which city is this? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(0, dtype=torch.float16)

        predictions = model.generate(**inputs, max_new_tokens=11)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(
            predictions[0].tolist(),
            [2, 45641, 35, 61, 343, 16, 42, 116, 31652, 35, 24, 18, 45, 10, 343, 6, 24, 18, 10, 4105, 50118],
        )
        self.assertEqual(generated_text, "Question: which city is this? Answer: it's not a city, it's a beach")

    @require_torch_multi_accelerator
    def test_inference_t5_multi_accelerator(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        device_map = device_map = {
            "query_tokens": 0,
            "vision_model": 0,
            "language_model": 1,
            "language_projection": 0,
            "qformer": 0,
        }

        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map=device_map
        )

        # prepare image
        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(f"{torch_device}:0", dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(predictions[0].tolist(), [0, 2335, 1556, 28, 1782, 30, 8, 2608, 1])
        self.assertEqual("woman playing with dog on the beach", generated_text)

        # image and context
        prompt = "Question: which city is this? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(f"{torch_device}:0", dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(
            predictions[0].tolist(),
            [0, 3, 7, 152, 67, 839, 1],
        )
        self.assertEqual(generated_text, "san diego")

    def test_expansion_in_processing(self):
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(torch_device)

        image = prepare_img()
        prompt = "Question: which city is this? Answer:"

        # Make sure we will go the legacy path by setting these args to None
        processor.num_query_tokens = None
        model.config.image_token_index = None
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs, do_sample=False, max_new_tokens=15)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Add args to the config to trigger new logic when inputs are expanded in processing file
        processor.num_query_tokens = model.config.num_query_tokens
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        model.config.image_token_index = len(processor.tokenizer) - 1
        model.resize_token_embeddings(processor.tokenizer.vocab_size, pad_to_multiple_of=64)

        # Generate again with new inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)
        predictions_expanded = model.generate(**inputs, do_sample=False, max_new_tokens=15)
        generated_text_expanded = processor.batch_decode(predictions_expanded, skip_special_tokens=True)[0].strip()

        self.assertTrue(generated_text_expanded == generated_text)

    @require_torch_accelerator
    def test_inference_itm(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name).to(torch_device)

        image = prepare_img()
        text = "A woman and her dog sitting in a beach"
        inputs = processor(images=image, text=text, return_tensors="pt").to(torch_device)

        # forward pass
        out_itm = model(**inputs, use_image_text_matching_head=True)
        out = model(**inputs)

        # verify
        expected_scores = torch.Tensor([[0.0238, 0.9762]])
        torch.testing.assert_close(torch.nn.Softmax()(out_itm[0].cpu()), expected_scores, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out[0].cpu(), torch.Tensor([[0.4406]]), rtol=1e-3, atol=1e-3)

    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_itm_fp16(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name, torch_dtype=torch.float16).to(torch_device)

        image = prepare_img()
        text = "A woman and her dog sitting in a beach"
        inputs = processor(images=image, text=text, return_tensors="pt").to(torch_device, dtype=torch.float16)

        # forward pass
        out_itm = model(**inputs, use_image_text_matching_head=True)
        out = model(**inputs)

        # verify
        expected_scores = torch.Tensor([[0.0239, 0.9761]])
        torch.testing.assert_close(torch.nn.Softmax()(out_itm[0].cpu().float()), expected_scores, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out[0].cpu().float(), torch.Tensor([[0.4406]]), rtol=1e-3, atol=1e-3)

    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_vision_with_projection_fp16(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2VisionModelWithProjection.from_pretrained(model_name, torch_dtype=torch.float16).to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        # forward pass
        out = model(**inputs)

        # verify
        expected_image_embeds = [
            -0.093994140625,
            -0.075927734375,
            0.031890869140625,
            0.053009033203125,
            0.0352783203125,
            -0.01190185546875,
        ]
        self.assertTrue(np.allclose(out.image_embeds[0][0][:6].tolist(), expected_image_embeds, atol=1e-3))

    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_text_with_projection_fp16(self):
        model_name = "Salesforce/blip2-itm-vit-g"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2TextModelWithProjection.from_pretrained(model_name, torch_dtype=torch.float16).to(torch_device)

        inputs = processor(text="a woman sitting on the beach with a dog", padding=True, return_tensors="pt").to(
            torch_device
        )

        # forward pass
        out = model(**inputs)

        # verify
        expected_text_embeds = [
            -0.1082763671875,
            0.053192138671875,
            -0.02825927734375,
            0.0169830322265625,
            0.08648681640625,
            -0.04656982421875,
        ]
        self.assertTrue(np.allclose(out.text_embeds[0][0][:6].tolist(), expected_text_embeds, atol=1e-3))
