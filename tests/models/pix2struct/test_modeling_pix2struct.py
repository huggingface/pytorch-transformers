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
"""Testing suite for the PyTorch Pix2Struct model."""

import copy
import inspect
import os
import tempfile
import unittest

import numpy as np
import requests

from transformers import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
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
    from torch import nn

    from transformers import (
        Pix2StructForConditionalGeneration,
        Pix2StructProcessor,
        Pix2StructTextModel,
        Pix2StructVisionModel,
    )


if is_vision_available():
    from PIL import Image


class Pix2StructVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=12,
        patch_embed_hidden_size=12,
        projection_dim=32,
        max_patches=64,
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
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.max_patches = max_patches
        self.seq_length = self.max_patches
        self.patch_proj_dim = ((patch_size**2) * num_channels) + 2

        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        flattened_patches = floats_tensor([self.batch_size, self.max_patches, self.patch_proj_dim])
        config = self.get_config()

        return config, flattened_patches

    def get_config(self):
        return Pix2StructVisionConfig(
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
            patch_embed_hidden_size=self.patch_embed_hidden_size,
        )

    def create_and_check_model(self, config, flattened_patches):
        model = Pix2StructVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(flattened_patches)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, flattened_patches = config_and_inputs
        inputs_dict = {
            "flattened_patches": flattened_patches,
            "attention_mask": torch.randint(0, 2, (self.batch_size, self.max_patches)),
        }
        return config, inputs_dict


@require_torch
class Pix2StructVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Pix2Struct does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Pix2StructVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Pix2StructVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Pix2StructVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Pix2StructVision does not use inputs_embeds")
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

            expected_arg_names = ["flattened_patches"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Training is tested directly on `Pix2StructTextImageModelTest`")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is tested directly on `Pix2StructTextImageModelTest`")
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

    @unittest.skip(reason="Training is tested directly on `Pix2StructTextImageModelTest`")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Pix2StructVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Pix2StructVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/pix2struct-textcaps-base"
        model = Pix2StructVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class Pix2StructTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=12,
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
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.d_kv = hidden_size // num_attention_heads
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
        return Pix2StructTextConfig(
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
            d_kv=self.d_kv,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Pix2StructTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Pix2StructTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Pix2StructTextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Pix2StructTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Pix2StructTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Training is tested directly on `Pix2StructTextImageModelTest`")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is tested directly on `Pix2StructTextImageModelTest`")
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

    @unittest.skip(reason="Pix2Struct does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Pix2StructTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Pix2StructTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/pix2struct-textcaps-base"
        model = Pix2StructTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class Pix2StructModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Pix2StructTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Pix2StructVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length  # need seq_length for common tests
        self.is_training = is_training
        self.max_patches = self.vision_model_tester.max_patches

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, flattened_patches = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config(text_config, vision_config)

        return config, input_ids, attention_mask, flattened_patches

    def get_config(self, text_config, vision_config):
        return Pix2StructConfig.from_text_vision_configs(text_config, vision_config, projection_dim=64)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, decoder_attention_mask, flattened_patches = config_and_inputs

        attention_mask = (flattened_patches.sum(dim=-1) != 0).float()

        inputs_dict = {
            "decoder_input_ids": input_ids,
            "labels": input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "flattened_patches": flattened_patches,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Pix2StructModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Pix2StructForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (Pix2StructForConditionalGeneration,) if is_torch_available() else {}
    pipeline_model_mapping = (
        {"image-to-text": Pix2StructForConditionalGeneration, "image-text-to-text": Pix2StructForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = True
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Pix2StructModelTester(self)

    def test_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)

            output = model(**input_dict)
            self.assertEqual(
                output[1].shape,
                (
                    self.model_tester.vision_model_tester.batch_size,
                    self.model_tester.text_model_tester.seq_length,
                    self.model_tester.text_model_tester.vocab_size,
                ),
            )

    def test_generative_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_generative_model_classes:
            model = model_class(config).eval().to(torch_device)

            output = model.generate(**input_dict, use_cache=False, min_new_tokens=10, max_new_tokens=10)
            output_use_cache = model.generate(**input_dict, use_cache=True, min_new_tokens=10, max_new_tokens=10)

            torch.testing.assert_close(output, output_use_cache)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Pix2StructModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "flattened_patches",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "head_mask",
                "decoder_head_mask",
                "cross_attn_head_mask",
                "encoder_outputs",
                "past_key_values",
                "labels",
                "decoder_inputs_embeds",
                "use_cache",
            ]

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            # hardcode labels to be the same as input_ids
            inputs["labels"] = inputs["input_ids"]

            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            # hardcode labels to be the same as input_ids
            inputs["labels"] = inputs["input_ids"]

            loss = model(**inputs).loss
            loss.backward()

    # override as the `logit_scale` parameter initilization is different for Pix2Struct
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
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # overwrite because `vocab_size` is not an attribute of `Pix2StructConfig` but rather `Pix2StructTextConfig`
    def test_resize_tokens_embeddings(self):
        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to False")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.text_config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Decoder input ids should be clamped to the maximum size of the vocabulary
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    # overwrite because `vocab_size` is not an attribute of `Pix2StructConfig` but rather `Pix2StructTextConfig`
    def test_resize_embeddings_untied(self):
        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to False")

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untie embeddings")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Decoder input ids should be clamped to the maximum size of the vocabulary
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    @unittest.skip(reason="Pix2Struct doesn't use tied weights")
    def test_tied_model_weights_key_ignore(self):
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

            try:
                input_ids = inputs_dict["input_ids"]
                flattened_patches = inputs_dict["flattened_patches"]  # Pix2Struct needs flattened_patches
                traced_model = torch.jit.trace(model, (input_ids, flattened_patches))
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
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Pix2StructConfig and check if we can load Pix2StructVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Pix2StructVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Pix2StructConfig and check if we can load Pix2StructTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Pix2StructTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, seq_length):
        # overwrite because # pix2struct seq length depends on image inputs
        seq_length = self.model_tester.max_patches
        encoder_expected_shape = (batch_size, config.num_attention_heads, seq_length, seq_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, seq_length):
        # overwrite because # pix2struct seq length depends on image inputs
        seq_length = self.model_tester.max_patches
        encoder_expected_shape = (batch_size, seq_length, config.hidden_size)
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )


# We will verify our results on an image of a stop sign
def prepare_img():
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
@slow
class Pix2StructIntegrationTest(unittest.TestCase):
    def test_inference_image_captioning(self):
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base").to(torch_device)
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")
        image = prepare_img()

        # image only
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        self.assertEqual(
            processor.decode(predictions[0], skip_special_tokens=True), "A stop sign is on a street corner."
        )

    def test_batched_inference_image_captioning(self):
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base").to(torch_device)
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")
        image_1 = prepare_img()

        second_url = (
            "https://www.connollycove.com/wp-content/uploads/2019/06/temple-bar-dublin-world-famous-irish-pub.jpg"
        )
        image_2 = Image.open(requests.get(second_url, stream=True).raw)

        # image only
        inputs = processor(images=[image_1, image_2], return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        self.assertEqual(
            processor.decode(predictions[0], skip_special_tokens=True), "A stop sign is on a street corner."
        )

        self.assertEqual(
            processor.decode(predictions[1], skip_special_tokens=True),
            "A row of books including The Temple Bar and Guiness.",
        )

    def test_batched_inference_image_captioning_conditioned(self):
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base").to(torch_device)
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")
        image_1 = prepare_img()

        second_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/temple-bar-dublin-world-famous-irish-pub.jpg"
        image_2 = Image.open(requests.get(second_url, stream=True).raw)
        texts = ["A picture of", "An photography of"]

        # image only
        inputs = processor(images=[image_1, image_2], text=texts, return_tensors="pt", add_special_tokens=False).to(
            torch_device
        )

        predictions = model.generate(**inputs)

        self.assertEqual(
            processor.decode(predictions[0], skip_special_tokens=True),
            "A picture of a stop sign with a red stop sign",
        )

        self.assertEqual(
            processor.decode(predictions[1], skip_special_tokens=True),
            "An photography of the Temple Bar and other places in the city.",
        )

    def test_vqa_model(self):
        model_id = "google/pix2struct-ai2d-base"

        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)

        model = Pix2StructForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(
            torch_device
        )
        processor = Pix2StructProcessor.from_pretrained(model_id)

        # image only
        text = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

        inputs = processor(images=image, return_tensors="pt", text=text).to(torch_device, torch.bfloat16)

        predictions = model.generate(**inputs)
        self.assertEqual(processor.decode(predictions[0], skip_special_tokens=True), "ash cloud")

    def test_vqa_model_batched(self):
        model_id = "google/pix2struct-ai2d-base"

        image_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo-2.png",
        ]

        images = [Image.open(requests.get(image_url, stream=True).raw) for image_url in image_urls]

        texts = [
            "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
            "What is the producer in the diagram? (1) Phytoplankton (2) Zooplankton (3) Large fish (4) Small fish",
        ]

        model = Pix2StructForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(
            torch_device
        )
        processor = Pix2StructProcessor.from_pretrained(model_id)

        inputs = processor(images=images, return_tensors="pt", text=texts).to(torch_device, torch.bfloat16)

        predictions = model.generate(**inputs)
        self.assertEqual(processor.decode(predictions[0], skip_special_tokens=True), "ash cloud")
        self.assertEqual(processor.decode(predictions[1], skip_special_tokens=True), "Phytoplankton")
