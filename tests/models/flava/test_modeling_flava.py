# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch FLAVA model."""

import inspect
import os
import random
import tempfile
import unittest

import numpy as np
import requests

from transformers import (
    FlavaConfig,
    FlavaImageCodebookConfig,
    FlavaImageConfig,
    FlavaMultimodalConfig,
    FlavaTextConfig,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

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
        FlavaForPreTraining,
        FlavaImageCodebook,
        FlavaImageModel,
        FlavaModel,
        FlavaMultimodalModel,
        FlavaTextModel,
    )
else:
    FlavaModel = None
    FlavaForPreTraining = None
    torch = {}


if is_vision_available():
    from PIL import Image

    from transformers import FlavaProcessor


class FlavaImageModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        qkv_bias=True,
        mask_token=True,
        vocab_size=99,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.mask_token = mask_token
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        num_patches = self.image_size // self.patch_size
        bool_masked_pos = (
            torch.rand((self.batch_size, num_patches, num_patches), device=pixel_values.device) < 0.9
        ).long()
        config = self.get_config()
        return config, pixel_values, bool_masked_pos

    def get_config(self):
        return FlavaImageConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            qkv_bias=self.qkv_bias,
            mask_token=self.mask_token,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model(self, config, pixel_values, bool_masked_pos):
        model = FlavaImageModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values, bool_masked_pos)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, bool_masked_pos = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "bool_masked_pos": bool_masked_pos}
        return config, inputs_dict


@require_torch
class FlavaImageModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as FLAVA does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (FlavaImageModel,) if is_torch_available() else ()

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = FlavaImageModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FlavaImageConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Flava does not use input_ids")
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

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # in FLAVA, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.model_tester.image_size, self.model_tester.image_size)
        patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            # FLAVA has a different seq_length
            image_size = (self.model_tester.image_size, self.model_tester.image_size)
            patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            seq_length = num_patches + 1

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
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

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

    @unittest.skip(reason="FlavaImageModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    # skip this test as FlavaImageModel has no base class and is
    # not available in MODEL_MAPPING
    @unittest.skip(reason="FlavaImageModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/flava-full"
        model = FlavaImageModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class FlavaTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=102,
        type_vocab_size=2,
        max_position_embeddings=512,
        position_embedding_type="absolute",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        qkv_bias=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.pad_token_id = pad_token_id

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

        token_type_ids = None

        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask

    def get_config(self):
        return FlavaTextConfig(
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            position_embedding_type=self.position_embedding_type,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            qkv_bias=self.qkv_bias,
        )

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask):
        model = FlavaTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class FlavaTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlavaTextModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = FlavaTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FlavaTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

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

    @unittest.skip(reason="FLAVA does not use input_embeds")
    def test_inputs_embeds(self):
        # FLAVA does not use inputs_embeds
        pass

    @unittest.skip(reason="FlavaTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="FlavaTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/flava-full"
        model = FlavaTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class FlavaMultimodalModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=44,
        use_input_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        ce_ignore_index=-100,
        use_cls_token=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_input_mask = use_input_mask
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.ce_ignore_index = ce_ignore_index
        self.use_cls_token = use_cls_token

    def prepare_config_and_inputs(self):
        hidden_states = floats_tensor([self.batch_size, self.seq_length - 1, self.hidden_size])

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

        return config, hidden_states, input_mask

    def get_config(self):
        return FlavaMultimodalConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            qkv_bias=self.qkv_bias,
            use_cls_token=self.use_cls_token,
            ce_ignore_index=self.ce_ignore_index,
        )

    def create_and_check_model(self, config, hidden_states, input_mask):
        model = FlavaMultimodalModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(hidden_states, attention_mask=input_mask)
            result = model(hidden_states)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, hidden_states, input_mask = config_and_inputs
        inputs_dict = {"hidden_states": hidden_states, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class FlavaMultimodalModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlavaMultimodalModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = FlavaMultimodalModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=FlavaMultimodalConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

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

            expected_arg_names = ["hidden_states"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @unittest.skip("FLAVA does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

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

    @unittest.skip(reason="FLAVA does not use input_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="FlavaMultimodalModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="FlavaMultimodalModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/flava-full"
        model = FlavaMultimodalModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class FlavaImageCodebookTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=112,
        num_channels=3,
        hidden_size=32,
        num_groups=2,
        vocab_size=99,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return FlavaImageCodebookConfig(
            hidden_size=self.hidden_size, num_groups=self.num_groups, vocab_size=self.vocab_size
        )

    def create_and_check_model(self, config, pixel_values):
        model = FlavaImageCodebook(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(
            result.shape, (self.batch_size, config.vocab_size, self.image_size // 8, self.image_size // 8)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class FlavaImageCodebookTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlavaImageCodebook,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    test_torchscript = False
    has_attentions = False

    def setUp(self):
        self.model_tester = FlavaImageCodebookTester(self)
        self.config_tester = ConfigTester(self, config_class=FlavaImageCodebookConfig, has_text_modality=False)

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

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @unittest.skip(reason="Flava does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="No embedding in multimodal model")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip
    def test_training(self):
        pass

    @unittest.skip
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="FlavaImageCodebook has no attentions")
    def test_retain_grad_hidden_states_attentions(self):
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

    @unittest.skip(reason="FLAVA does not use input_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="FlavaImageCodebook has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="FlavaImageCodebook has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/flava-full"
        model = FlavaImageCodebook.from_pretrained(model_name)
        self.assertIsNotNone(model)


class FlavaModelTester:
    model_class = FlavaModel

    def __init__(
        self,
        parent,
        text_kwargs=None,
        image_kwargs=None,
        multimodal_kwargs=None,
        image_codebook_kwargs=None,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if image_kwargs is None:
            image_kwargs = {}
        if multimodal_kwargs is None:
            multimodal_kwargs = {}
        if image_codebook_kwargs is None:
            image_codebook_kwargs = {}

        self.parent = parent
        self.image_model_tester = FlavaImageModelTester(parent, **image_kwargs)
        self.text_model_tester = FlavaTextModelTester(parent, **text_kwargs)
        self.multimodal_model_tester = FlavaMultimodalModelTester(parent, **multimodal_kwargs)
        self.image_codebook_tester = FlavaImageCodebookTester(parent, **image_codebook_kwargs)
        self.is_training = is_training
        self.config_tester = ConfigTester(self, config_class=FlavaConfig, hidden_size=37)
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test

    def test_config(self):
        self.config_tester.run_common_tests()

    def prepare_config_and_inputs_for_common(self):
        _, pixel_values, bool_masked_pos = self.image_model_tester.prepare_config_and_inputs()
        _, input_ids, token_type_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
        }

    def get_config(self):
        return FlavaConfig.from_configs(
            self.image_model_tester.get_config(),
            self.text_model_tester.get_config(),
            self.multimodal_model_tester.get_config(),
            self.image_codebook_tester.get_config(),
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
        )

    def create_and_check_model(self, config, inputs):
        self._test_model(config, inputs, test_image=True)
        self._test_model(config, inputs, test_text=True)
        self._test_model(config, inputs, test_image=True, test_text=True)

    def _test_model(self, config, inputs, test_image=False, test_text=False):
        model = self.model_class(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                input_ids=inputs["input_ids"] if test_text else None,
                attention_mask=inputs["attention_mask"] if test_text else None,
                token_type_ids=inputs["token_type_ids"] if test_text else None,
                pixel_values=inputs["pixel_values"] if test_image else None,
                bool_masked_pos=inputs["bool_masked_pos"] if test_image else None,
            )
        image_size = (self.image_model_tester.image_size, self.image_model_tester.image_size)
        patch_size = (self.image_model_tester.patch_size, self.image_model_tester.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        if test_image:
            self.parent.assertEqual(
                result.image_embeddings.shape,
                (self.image_model_tester.batch_size, num_patches + 1, self.image_model_tester.hidden_size),
            )
        else:
            self.parent.assertIsNone(result.image_embeddings)

        if test_text:
            self.parent.assertEqual(
                result.text_embeddings.shape,
                (
                    self.text_model_tester.batch_size,
                    self.text_model_tester.seq_length,
                    self.text_model_tester.hidden_size,
                ),
            )
        else:
            self.parent.assertIsNone(result.text_embeddings)

        if test_image and test_text:
            self.parent.assertEqual(
                result.multimodal_embeddings.shape,
                (
                    self.multimodal_model_tester.batch_size,
                    self.text_model_tester.seq_length + num_patches + 2,
                    self.multimodal_model_tester.hidden_size,
                ),
            )
        else:
            self.parent.assertIsNone(result.multimodal_embeddings)


@require_torch
class FlavaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FlavaModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": FlavaModel} if is_torch_available() else {}
    class_for_tester = FlavaModelTester
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = self.class_for_tester(self)
        common_properties = ["projection_dim", "logit_scale_init_value", "init_codebook"]
        self.config_tester = ConfigTester(
            self, config_class=FlavaConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="FlavaModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override as the `logit_scale` parameter initilization is different for FLAVA
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale" or name == "flava.logit_scale":
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

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        configs_no_init.return_loss = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # FLAVA needs pixel_values

                if "input_ids_masked" in inputs_dict:
                    # For pretraining
                    inputs = (input_ids, inputs_dict["input_ids_masked"], pixel_values)
                else:
                    inputs = (input_ids, pixel_values)

                traced_model = torch.jit.trace(model, inputs)
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
            # Non persistent buffers won't be in original state dict
            loaded_model_state_dict.pop("text_model.embeddings.token_type_ids", None)

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

    def test_load_image_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save FlavaConfig and check if we can load FlavaImageConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            image_config = FlavaImageConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.image_config.to_dict(), image_config.to_dict())

        # Save FlavaConfig and check if we can load FlavaTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = FlavaTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

        # Save FlavaConfig and check if we can load FlavaMultimodalConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            multimodal_config = FlavaMultimodalConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.multimodal_config.to_dict(), multimodal_config.to_dict())

    # overwrite from common since FlavaModel/TFFlavaModel return FLAVAOutput/TFFLAVAOutput
    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/flava-full"
        model = FlavaModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class FlavaForPreTrainingTester(FlavaModelTester):
    model_class = FlavaForPreTraining

    def prepare_config_and_inputs_for_common(self):
        _, pixel_values, bool_masked_pos = self.image_model_tester.prepare_config_and_inputs()
        _, input_ids, token_type_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        config = self.get_config()

        input_ids_masked = input_ids.detach().clone()
        input_ids_masked[:, 1:3] = 100
        mlm_labels = input_ids.detach().clone()
        mlm_labels[:, :] = config.ce_ignore_index
        mlm_labels[:, 1:3] = input_ids[:, 1:3]
        mim_labels = torch.randint(
            0, self.image_model_tester.vocab_size, bool_masked_pos.size(), device=bool_masked_pos.device
        ).long()
        mim_labels[bool_masked_pos.ne(True)] = config.ce_ignore_index
        itm_labels = torch.ones(mlm_labels.size(0), device=bool_masked_pos.device).long()

        return config, {
            "input_ids": input_ids,
            "input_ids_masked": input_ids_masked,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
            "mlm_labels": mlm_labels,
            "mim_labels": mim_labels,
            "itm_labels": itm_labels,
            "return_loss": True,
        }

    def _test_model(self, config, inputs, test_image=False, test_text=False):
        model = self.model_class(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                input_ids=inputs["input_ids"] if test_text else None,
                input_ids_masked=inputs["input_ids_masked"] if test_text else None,
                attention_mask=inputs["attention_mask"] if test_text else None,
                token_type_ids=inputs["token_type_ids"] if test_text else None,
                pixel_values=inputs["pixel_values"] if test_image else None,
                bool_masked_pos=inputs["bool_masked_pos"] if test_image else None,
                mlm_labels=inputs["mlm_labels"],
                mim_labels=inputs["mim_labels"],
                itm_labels=inputs["itm_labels"],
                return_loss=inputs["return_loss"],
            )
        image_size = (self.image_model_tester.image_size, self.image_model_tester.image_size)
        patch_size = (self.image_model_tester.patch_size, self.image_model_tester.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        if test_image:
            self.parent.assertEqual(
                result.image_embeddings.shape,
                (self.image_model_tester.batch_size, num_patches + 1, self.image_model_tester.hidden_size),
            )
            if not test_text:
                self.parent.assertEqual(
                    result.loss_info.mim.dim(),
                    0,
                )
                self.parent.assertEqual(
                    result.mim_logits.shape,
                    (inputs["bool_masked_pos"].sum().item(), self.image_model_tester.vocab_size),
                )

        else:
            self.parent.assertIsNone(result.image_embeddings)

        if test_text:
            self.parent.assertEqual(
                result.text_embeddings.shape,
                (
                    self.text_model_tester.batch_size,
                    self.text_model_tester.seq_length,
                    self.text_model_tester.hidden_size,
                ),
            )
            if not test_image:
                self.parent.assertEqual(result.loss_info.mlm.dim(), 0)
                self.parent.assertEqual(
                    result.mlm_logits.shape,
                    (
                        (inputs["mlm_labels"] != self.multimodal_model_tester.ce_ignore_index).sum().item(),
                        self.text_model_tester.vocab_size,
                    ),
                )
        else:
            self.parent.assertIsNone(result.text_embeddings)

        if test_image and test_text:
            self.parent.assertEqual(
                result.multimodal_masked_embeddings.shape,
                (
                    self.multimodal_model_tester.batch_size,
                    self.text_model_tester.seq_length + num_patches + 2,
                    self.multimodal_model_tester.hidden_size,
                ),
            )
            self.parent.assertEqual(
                result.itm_logits.shape,
                (self.text_model_tester.batch_size, 2),
            )
            self.parent.assertEqual(
                result.mmm_text_logits.shape,
                (
                    (inputs["mlm_labels"] != self.multimodal_model_tester.ce_ignore_index).sum().item(),
                    self.text_model_tester.vocab_size,
                ),
            )
            self.parent.assertEqual(
                result.mmm_image_logits.shape,
                (inputs["bool_masked_pos"].sum().item(), self.image_model_tester.vocab_size),
            )
            self.parent.assertEqual(
                result.contrastive_logits_per_image.shape,
                (self.image_model_tester.batch_size, self.text_model_tester.batch_size),
            )
            self.parent.assertEqual(
                result.contrastive_logits_per_text.shape,
                (self.text_model_tester.batch_size, self.image_model_tester.batch_size),
            )

            for item in [
                result.loss_info.global_contrastive,
                result.loss_info.itm,
                result.loss_info.mmm_text,
                result.loss_info.mmm_image,
            ]:
                self.parent.assertEqual(item.dim(), 0)

            for item in [result.loss_info.mim, result.loss_info.mlm]:
                self.parent.assertIsNone(item)

        else:
            self.parent.assertIsNone(result.multimodal_masked_embeddings)
            for item in [
                result.loss_info.global_contrastive,
                result.loss_info.itm,
                result.loss_info.mmm_text,
                result.loss_info.mmm_image,
            ]:
                self.parent.assertIsNone(item)

        self.parent.assertIsNone(result.multimodal_embeddings)


@require_torch
class FlavaForPreTrainingTest(FlavaModelTest):
    all_model_classes = (FlavaForPreTraining,) if is_torch_available() else ()
    class_for_tester = FlavaForPreTrainingTester
    test_torchscript = False

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
class FlavaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "facebook/flava-full"
        model = FlavaModel.from_pretrained(model_name).to(torch_device)
        processor = FlavaProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=[image, image],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        # verify the embeddings
        self.assertAlmostEqual(outputs.image_embeddings.sum().item(), -1352.53540, places=4)
        self.assertAlmostEqual(outputs.text_embeddings.sum().item(), -198.98225, places=4)
        self.assertAlmostEqual(outputs.multimodal_embeddings.sum().item(), -4030.4604492, places=4)


@require_vision
@require_torch
class FlavaForPreTrainingIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "facebook/flava-full"
        model = FlavaForPreTraining.from_pretrained(model_name).to(torch_device)
        processor = FlavaProcessor.from_pretrained(model_name)
        torch.manual_seed(1)
        random.seed(1)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=[image, image],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
            return_codebook_pixels=True,
            return_image_mask=True,
        )
        # Create a clone of the input_ids tensor that will be its masked version
        inputs["input_ids_masked"] = inputs["input_ids"].clone()
        # Mask the tokens "a" & "cat" from the "a photo of a cat" text using the special 103 value
        inputs["input_ids_masked"][0, 4:6] = 103
        # MLM labels. It is a cloned version of input_ids where all values are -100 (i.e., ignored)
        # except those that are masked, whose original values are stored
        inputs["mlm_labels"] = inputs["input_ids"].clone()
        inputs["mlm_labels"][:, :] = -100
        inputs["mlm_labels"][0, 4:6] = inputs["input_ids"][0, 4:6]
        inputs = inputs.to(torch_device)
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.contrastive_logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.contrastive_logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[16.1291, 8.4033], [16.1291, 8.4033]], device=torch_device)
        torch.testing.assert_close(outputs.contrastive_logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)
        self.assertAlmostEqual(outputs.loss_info.mmm_text.item(), 2.0727925, places=4)
        self.assertAlmostEqual(outputs.loss_info.mmm_image.item(), 7.0282096, places=4)
        self.assertAlmostEqual(outputs.loss.item(), 11.3792324, places=4)

    @slow
    def test_inference_with_itm_labels(self):
        model_name = "facebook/flava-full"
        model = FlavaForPreTraining.from_pretrained(model_name).to(torch_device)
        processor = FlavaProcessor.from_pretrained(model_name)
        torch.manual_seed(1)
        random.seed(1)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=[image, image],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
            return_codebook_pixels=True,
            return_image_mask=True,
        )
        # Create a clone of the input_ids tensor that will be its masked version
        inputs["input_ids_masked"] = inputs["input_ids"].clone()
        # Mask the tokens "a" & "cat" from the "a photo of a cat" text using the special 103 value
        inputs["input_ids_masked"][0, 4:6] = 103
        # MLM labels. It is a cloned version of input_ids where all values are -100 (i.e., ignored)
        # except those that are masked, whose original values are stored
        inputs["mlm_labels"] = inputs["input_ids"].clone()
        inputs["mlm_labels"][:, :] = -100
        inputs["mlm_labels"][0, 4:6] = inputs["input_ids"][0, 4:6]
        # Manually create the itm_labels tensor that indicates if the image-text match.
        # In this case, the firs pair matches and the second does not
        inputs["itm_labels"] = torch.tensor([1, 0])
        inputs = inputs.to(torch_device)
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.contrastive_logits_per_image.shape,
            torch.Size((torch.count_nonzero(inputs["itm_labels"]).item(), inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.contrastive_logits_per_text.shape,
            torch.Size((torch.count_nonzero(inputs["itm_labels"]).item(), inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[16.1291, 8.4033], [16.1291, 8.4033]], device=torch_device)
        torch.testing.assert_close(outputs.contrastive_logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)
        self.assertAlmostEqual(outputs.loss_info.mmm_text.item(), 2.0727925, places=4)
        self.assertAlmostEqual(outputs.loss_info.mmm_image.item(), 6.8965902, places=4)
        self.assertAlmostEqual(outputs.loss.item(), 9.6084213, places=4)
