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
"""Testing suite for the PyTorch InstructBLIP model."""

import inspect
import tempfile
import unittest

import numpy as np
import pytest
import requests
from parameterized import parameterized

from transformers import (
    CONFIG_MAPPING,
    InstructBlipConfig,
    InstructBlipProcessor,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
)
from transformers.testing_utils import (
    require_accelerate,
    require_bitsandbytes,
    require_torch,
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
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch
    from torch import nn

    from transformers import InstructBlipForConditionalGeneration, InstructBlipVisionModel


if is_vision_available():
    from PIL import Image


class InstructBlipVisionModelTester:
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

        # in case of a vision transformer, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return InstructBlipVisionConfig(
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
        model = InstructBlipVisionModel(config=config)
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
class InstructBlipVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as InstructBLIP's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (InstructBlipVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = InstructBlipVisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=InstructBlipConfig,
            has_text_modality=False,
            common_properties=["num_query_tokens", "image_token_index"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="InstructBLIP's vision encoder does not use inputs_embeds")
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

    @unittest.skip(reason="InstructBlipVisionModel is an internal building block, doesn't support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="InstructBlipVisionModel is an internal building block, doesn't support standalone training")
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

    @unittest.skip(reason="InstructBlipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="InstructBlipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/instructblip-flan-t5-xl"
        model = InstructBlipVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class InstructBlipQFormerModelTester:
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        qformer_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
            qformer_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask, qformer_input_ids, qformer_attention_mask

    def get_config(self):
        return InstructBlipQFormerConfig(
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
        )


# this class is based on `OPTModelTester` found in tests/models/opt/test_modeling_opt.py
class InstructBlipTextModelDecoderOnlyTester:
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
        max_position_embeddings=100,
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
class InstructBlipForConditionalGenerationDecoderOnlyModelTester:
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
        self.vision_model_tester = InstructBlipVisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = InstructBlipQFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = InstructBlipTextModelDecoderOnlyTester(parent, **text_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length + num_query_tokens  # need seq_length for common tests
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens
        self.image_token_index = image_token_index

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        _, _, _, qformer_input_ids, qformer_attention_mask = self.qformer_model_tester.prepare_config_and_inputs()
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()
        vision_tokens = (
            torch.ones((input_ids.shape[0], self.num_query_tokens), device=torch_device, dtype=input_ids.dtype)
            * self.image_token_index
        )
        input_ids[input_ids == self.image_token_index] = self.text_model_tester.pad_token_id
        input_ids = torch.cat([vision_tokens, input_ids], dim=-1)
        vision_attention_mask = torch.ones_like(vision_tokens)
        attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=-1)

        return config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values

    def get_config(self):
        return InstructBlipConfig.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
            image_token_index=self.image_token_index,
        )

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values
    ):
        model = InstructBlipForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                qformer_input_ids=qformer_input_ids,
                qformer_attention_mask=qformer_attention_mask,
            )

        expected_seq_length = self.num_query_tokens + self.text_model_tester.seq_length
        self.parent.assertEqual(
            result.logits.shape,
            (self.vision_model_tester.batch_size, expected_seq_length, self.text_model_tester.vocab_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "qformer_input_ids": qformer_input_ids,
            "qformer_attention_mask": qformer_attention_mask,
            "labels": input_ids,
        }
        return config, inputs_dict


@require_torch
class InstructBlipForConditionalGenerationDecoderOnlyTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (InstructBlipForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (InstructBlipForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": InstructBlipForConditionalGeneration}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = True
    test_attention_outputs = False
    test_torchscript = False
    _is_composite = True

    def setUp(self):
        self.model_tester = InstructBlipForConditionalGenerationDecoderOnlyModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=InstructBlipConfig,
            has_text_modality=False,
            common_properties=["num_query_tokens", "image_token_index"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="InstructBlipForConditionalGeneration doesn't support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Tied weights are tested in individual model tests")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="InstructBlipModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="There's no base InstructBlipModel")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base InstructBlipModel")
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
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save InstructBlipConfig and check if we can load InstructBlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = InstructBlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save InstructBlipConfig and check if we can load InstructBlipQFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = InstructBlipQFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/instructblip-flan-t5-xl"
        model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
        self.assertIsNotNone(model)

    # overwrite because InstructBLIP internally calls LM.generate() with embeds thus it cannot operate in no cache format
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

    # overwrite because InstructBLIP cannot generate only from input ids, and requires `pixel` values and `qformer_input_ids` in all cases to be present
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
            qformer_input_ids = inputs_dict["qformer_input_ids"]
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(
                **model_kwargs, pixel_values=pixel_values, qformer_input_ids=qformer_input_ids
            ).logits[:, -1, :]

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
            next_logits_with_padding = model(
                **model_kwargs, pixel_values=pixel_values, qformer_input_ids=qformer_input_ids
            ).logits[:, -1, :]

            # They should result in very similar logits
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=1e-5))

    @unittest.skip(
        "InstructBLIP cannot generate only from input ids, and requires pixel values in all cases to be present"
    )
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    def test_generate_from_inputs_embeds(self, _, num_beams):
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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_torch
@slow
class InstructBlipModelIntegrationTest(unittest.TestCase):
    @require_bitsandbytes
    @require_accelerate
    def test_inference_vicuna_7b(self):
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b", load_in_8bit=True, low_cpu_mem_usage=True
        )

        url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "What is unusual about this image?"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, torch.float16)

        # verify generation
        outputs = model.generate(**inputs, max_new_tokens=30)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        expected_outputs = [32001] * 32 + [2, 1724, 338, 22910, 1048, 445, 1967, 29973, 450, 22910, 9565, 310, 445, 1967, 338, 393, 263, 767, 338, 13977, 292, 22095, 373, 278, 1250, 310, 263, 13328, 20134, 29963, 1550, 19500, 373, 263, 19587, 4272, 11952, 29889]  # fmt: off

        self.assertEqual(outputs[0].tolist(), expected_outputs)
        self.assertEqual(
            generated_text,
            "What is unusual about this image? The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV while driving on a busy city street.",
        )

    def test_inference_flant5_xl(self):
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(torch_device)

        url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "What is unusual about this image?"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device)

        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        expected_outputs = [0, 37, 1023, 9850, 7, 3, 9, 388, 3575, 53, 4954, 30, 8, 223, 13, 3, 9, 4459, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 37, 388, 19, 5119, 3, 9, 4459, 8677, 28, 3, 9, 2756, 4459, 6177, 6, 11, 3, 88, 19, 338, 46, 3575, 53, 1476, 12, 743, 112, 2491, 5, 37, 1023, 19, 7225, 788, 12, 8, 685, 24, 34, 1267, 3, 9, 388, 3575, 53, 4954, 30, 8, 223, 13, 3, 9, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 94, 19, 487, 24, 8, 388, 19, 1119, 12, 1097, 540, 57, 692, 112, 10428, 30, 8, 223, 13, 8, 4049, 6, 68, 34, 19, 92, 487, 24, 3, 88, 19, 1119, 12, 1097, 97, 57, 692, 112, 10428, 30, 8, 223, 13, 8, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 3, 13865, 13, 8, 1053, 21, 8, 388, 31, 7, 2874, 6, 34, 19, 964, 24, 3, 88, 19, 1119, 12, 1097, 97, 57, 692, 112, 10428, 30, 8, 223, 13, 8, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 1]  # fmt: skip

        expected_outputs = [0, 37, 7225, 1023, 9850, 7, 3, 9, 388, 3575, 53, 4954, 30, 8, 223, 13, 3, 9, 4459, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 37, 388, 19, 5119, 3, 9, 4459, 8677, 28, 46, 3575, 53, 1476, 5223, 12, 34, 6, 15495, 24, 3, 88, 19, 692, 112, 293, 10428, 44, 234, 1066, 145, 338, 3, 9, 50, 1106, 3522, 144, 42, 2192, 7919, 31, 7, 5, 37, 1023, 92, 1267, 3, 9, 381, 13, 119, 3203, 16, 8, 2458, 6, 379, 14264, 6, 9256, 7, 6, 11, 11718, 7, 5, 1]  # fmt: skip

        self.assertEqual(outputs[0].tolist(), expected_outputs)
        self.assertEqual(
            generated_text,
            "The unusual image depicts a man ironing clothes on the back of a yellow van in the middle of a busy city street. The man is wearing a yellow shirt with an ironing board attached to it, suggesting that he is doing his own laundry at home rather than using a laundromat or dry cleaner's. The image also shows a number of other vehicles in the background, including buses, taxis, and motorcycles.",
        )

    def test_inference_interpolate_pos_encoding(self):
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(torch_device)
        processor.image_processor.size = {"height": 500, "width": 500}

        image = prepare_img()
        prompt = "What's in the image?"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs, interpolate_pos_encoding=True)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        self.assertEqual(
            predictions[0].tolist(), [0, 37, 1023, 753, 3, 9, 2335, 3823, 30, 8, 2608, 28, 3, 9, 1782, 5, 1]
        )
        self.assertEqual(generated_text, "The image features a woman sitting on the beach with a dog.")

    def test_expansion_in_processing(self):
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(torch_device)

        image = prepare_img()
        prompt = "What's in the image?"

        # Make sure we will go the legacy path by setting these args to None
        processor.num_query_tokens = None
        model.config.image_token_index = None
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs, do_sample=False, max_new_tokens=15)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Add args to the config to trigger new logic when inputs are expanded in processing file
        processor.num_query_tokens = model.config.num_query_tokens
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        model.config.image_token_index = len(processor.tokenizer) - 2
        model.resize_token_embeddings(processor.tokenizer.vocab_size, pad_to_multiple_of=64)

        # Generate again with new inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)
        predictions_expanded = model.generate(**inputs, do_sample=False, max_new_tokens=15)
        generated_text_expanded = processor.batch_decode(predictions_expanded, skip_special_tokens=True)[0].strip()

        self.assertTrue(generated_text_expanded == generated_text)
