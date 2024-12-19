# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the TensorFlow Whisper model."""

from __future__ import annotations

import inspect
import os
import tempfile
import traceback
import unittest

import numpy as np

from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor, WhisperProcessor
from transformers.testing_utils import (
    is_tf_available,
    require_read_token,
    require_tf,
    require_tokenizers,
    run_test_in_subprocess,
    slow,
)
from transformers.utils import cached_property
from transformers.utils.import_utils import is_datasets_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    import datasets
    from datasets import load_dataset


if is_tf_available():
    import tensorflow as tf

    from transformers import TFWhisperForConditionalGeneration, TFWhisperModel, set_seed
    from transformers.models.whisper.modeling_tf_whisper import (
        TFWhisperDecoder,
        TFWhisperEncoder,
        sinusoidal_embedding_init,
    )


def prepare_whisper_inputs_dict(
    config,
    input_features,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.where(decoder_input_ids != config.pad_token_id, 1, 0)
    if head_mask is None:
        head_mask = tf.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


@require_tf
class TFWhisperModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=60,
        is_training=True,
        use_labels=False,
        vocab_size=200,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        input_channels=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=30,
        max_target_positions=60,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
        num_mel_bins=80,
        decoder_start_token_id=85,
        num_conv_layers=1,
        suppress_tokens=None,
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
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.num_conv_layers = num_conv_layers
        self.suppress_tokens = suppress_tokens

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.num_mel_bins, self.seq_length], self.vocab_size)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_whisper_inputs_dict(
            config,
            attention_mask=None,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
        return config, inputs_dict

    def get_config(self):
        return WhisperConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_ffn_dim=self.hidden_size,
            encoder_ffn_dim=self.hidden_size,
            decoder_start_token_id=self.decoder_start_token_id,
            suppress_tokens=self.suppress_tokens,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def create_and_check_model_forward(self, config, inputs_dict):
        model = TFWhisperModel(config=config)

        input_features = inputs_dict["input_features"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state

        self.parent.assertTrue(last_hidden_state.shape, (13, 7, 16))

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFWhisperModel(config=config).get_decoder()
        # take a slice so we're shorter than the seqeuence length and can append later
        input_ids = inputs_dict["decoder_input_ids"][:, :-10]
        attention_mask = inputs_dict["decoder_attention_mask"][:, :-10]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_token = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_tokens = tf.where(next_token <= 2, 2, next_token)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = np.random.randint(0, output_from_past.shape[-1])
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(np.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = TFWhisperModel(config=config)
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = TFWhisperEncoder.from_pretrained(tmpdirname)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_features"])[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = TFWhisperDecoder.from_pretrained(tmpdirname)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max() < 1e-3)


@require_tf
class TFWhisperModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFWhisperModel, TFWhisperForConditionalGeneration) if is_tf_available() else ()
    all_generative_model_classes = (TFWhisperForConditionalGeneration,) if is_tf_available() else ()
    pipeline_model_mapping = {"feature-extraction": TFWhisperModel} if is_tf_available() else {}
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    test_onnx = False

    input_name = "input_features"

    # TODO (ydshieh): undo skip once a fix is done on TF side.
    @unittest.skip("Skip for now as TF 2.13 breaks it on GPU")
    def test_xla_generate_slow(self):
        super().test_xla_generate_slow()

    def setUp(self):
        self.model_tester = TFWhisperModelTester(self)
        self.config_tester = ConfigTester(self, config_class=WhisperConfig)
        self.maxDiff = 3000

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            model.build_in_name_scope()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_requires_grad_encoder_embed_positions(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            encoder = model.get_encoder()
            self.assertFalse(encoder.embed_positions.trainable)

    def test_encoder_sinusoidal_embed_positions(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.build_in_name_scope()

            embeds = model.get_encoder().embed_positions.get_weights()[0]
            sinusoids = sinusoidal_embedding_init(embeds.shape).numpy()
            self.assertTrue(np.allclose(embeds, sinusoids))

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        # cut to half length & take max batch_size 3
        max_batch_size = 3
        input_ids = input_ids[:max_batch_size, :, :]

        # generate max 3 tokens
        max_length = 4
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config.pad_token_id = config.eos_token_id

        return config, input_ids, None, max_length

    # not implemented currently
    def test_inputs_embeds(self):
        pass

    @unittest.skip("Training is not yet supported")
    def test_training(self):
        pass

    def test_generate_with_head_masking(self):
        pass

    @unittest.skip("fp16 is not yet supported for TF models")
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        config.max_target_positions = 400
        input_features = input_dict["input_features"]
        model = TFWhisperForConditionalGeneration(config)
        model.generate(input_features)
        model.generate(input_features, num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_features",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["decoder_position_ids", "head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model._get_feat_extract_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)

                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_length)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=5e-5, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as test recently became flaky
        super().check_pt_tf_outputs(tf_outputs, pt_outputs, model_class, tol, name, attributes)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", encoder_key_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)

            subsampled_encoder_seq_length = model._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model._get_feat_extract_output_lengths(encoder_key_length)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 5

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    subsampled_encoder_key_length,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

    def test_generate_without_input_ids(self):
        pass

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        batch_size, mel, seq_length = input_ids.shape
        subsampled_seq_length = self.model_tester.get_subsampled_output_lengths(seq_length)
        num_sequences_in_output = batch_size * num_return_sequences
        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # scores
        self._check_scores(num_sequences_in_output, output.scores, length=gen_len, config=config)

        # Attentions
        # encoder
        self._check_encoder_attention_for_generate(
            output.encoder_attentions, batch_size, config, subsampled_seq_length
        )
        # decoder
        self._check_attentions_for_generate(
            num_sequences_in_output,
            output.decoder_attentions,
            min_length=1,
            max_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

        # Hidden States
        # encoder
        self._check_encoder_hidden_states_for_generate(
            output.encoder_hidden_states, batch_size, config, subsampled_seq_length
        )

        # decoder
        self._check_hidden_states_for_generate(
            num_sequences_in_output,
            output.decoder_hidden_states,
            min_length=1,
            max_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

    # overwritten from parent due to the inability to work when non-text inputs are not passed AND because the input is
    # `input_features`
    def test_lm_head_model_random_no_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_features = inputs_dict.get("input_features", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined model needs input_features
                with self.assertRaises(AssertionError):
                    model.generate(do_sample=True, max_length=5)
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(input_features, do_sample=True))

            with self.assertRaises(ValueError):
                # generating multiple sequences when no beam search generation
                # is not allowed as it would always generate the same sequences
                model.generate(input_features, do_sample=False, num_return_sequences=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(model.generate(input_features, do_sample=True, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_features, do_sample=True, bad_words_ids=bad_words_ids, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_features.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    # overwritten from parent due to the inability to work when non-text inputs are not passed AND because the input is
    # `input_features`
    def test_lm_head_model_random_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_features = inputs_dict.get("input_features", None)

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined model needs input_ids, num_return_sequences = 1
                self._check_generated_ids(model.generate(input_features, do_sample=True, num_beams=2))

            with self.assertRaises(ValueError):
                # generating more sequences than having beams leads is not possible
                model.generate(input_features, do_sample=False, num_return_sequences=3, num_beams=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(
                model.generate(
                    input_features,
                    do_sample=True,
                    num_beams=2,
                    num_return_sequences=2,
                )
            )
            # num_return_sequences > 1, greedy
            self._check_generated_ids(
                model.generate(input_features, do_sample=False, num_beams=2, num_return_sequences=2)
            )

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_features, do_sample=False, bad_words_ids=bad_words_ids, num_beams=2, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_features.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_generate_with_prompt_ids_and_task_and_language(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TFWhisperForConditionalGeneration(config)
        input_features = input_dict["input_features"]
        prompt_ids = np.arange(5)
        language = "<|de|>"
        task = "translate"
        lang_id = 6
        task_id = 7
        model.generation_config.__setattr__("lang_to_id", {language: lang_id})
        model.generation_config.__setattr__("task_to_id", {task: task_id})

        output = model.generate(input_features, max_new_tokens=5, task=task, language=language, prompt_ids=prompt_ids)

        expected_output_start = [
            *prompt_ids.tolist(),
            model.generation_config.decoder_start_token_id,
            lang_id,
            task_id,
        ]
        for row in output.numpy().tolist():
            self.assertListEqual(row[: len(expected_output_start)], expected_output_start)

    def test_generate_with_prompt_ids_and_forced_decoder_ids(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TFWhisperForConditionalGeneration(config)
        input_features = input_dict["input_features"]
        prompt_ids = np.asarray(range(5))
        forced_decoder_ids = [(1, 6), (2, 7), (3, 8)]

        output = model.generate(
            input_features, max_new_tokens=5, forced_decoder_ids=forced_decoder_ids, prompt_ids=prompt_ids
        )

        expected_output_start = [
            *prompt_ids.tolist(),
            model.generation_config.decoder_start_token_id,
            *[token for _rank, token in forced_decoder_ids],
        ]
        for row in output.numpy().tolist():
            self.assertListEqual(row[: len(expected_output_start)], expected_output_start)


def _load_datasamples(num_samples):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # automatic decoding with librispeech
    speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

    return [x["array"] for x in speech_samples]


def _test_large_logits_librispeech(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        set_seed(0)

        model = TFWhisperModel.from_pretrained("openai/whisper-large")

        input_speech = _load_datasamples(1)

        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        processed_inputs = processor(
            audio=input_speech, text="This part of the speech", add_special_tokens=False, return_tensors="tf"
        )
        input_features = processed_inputs.input_features
        decoder_input_ids = processed_inputs.labels

        logits = model(
            input_features,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        logits = logits.last_hidden_state @ tf.transpose(model.model.decoder.embed_tokens.weights[0])

        # fmt: off
        EXPECTED_LOGITS = tf.convert_to_tensor(
            [
                2.1382, 0.9381, 4.4671, 3.5589, 2.4022, 3.8576, -0.6521, 2.5472,
                1.8301, 1.9957, 2.3432, 1.4678, 0.5459, 2.2597, 1.5179, 2.5357,
                1.1624, 0.6194, 1.0757, 1.8259, 2.4076, 1.6601, 2.3503, 1.3376,
                1.9891, 1.8635, 3.8931, 5.3699, 4.4772, 3.9184
            ]
        )
        # fmt: on

        unittest.TestCase().assertTrue(np.allclose(logits[0, 0, :30], EXPECTED_LOGITS, atol=1e-4))
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def _test_large_generation(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

        input_speech = _load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features

        generated_ids = model.generate(
            input_features,
            do_sample=False,
            max_length=20,
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " Mr. Quilter is the apostle of the middle classes and we are glad"
        unittest.TestCase().assertEqual(transcript, EXPECTED_TRANSCRIPT)
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def _test_large_generation_multilingual(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

        # update generation config
        generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v2")

        token = os.getenv("HF_HUB_READ_TOKEN", True)
        ds = load_dataset(
            "mozilla-foundation/common_voice_6_1",
            "ja",
            split="test",
            streaming=True,
            trust_remote_code=True,
            token=token,
        )
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        input_speech = next(iter(ds))["audio"]["array"]
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features

        generated_ids = model.generate(
            input_features,
            do_sample=False,
            max_length=20,
            language="<|ja|>",
            task="transcribe",
            generation_config=generation_config,
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "木村さんに電話を貸してもらいました"
        unittest.TestCase().assertEqual(transcript, EXPECTED_TRANSCRIPT)

        generated_ids = model.generate(
            input_features,
            do_sample=False,
            max_length=20,
            language="<|en|>",
            task="transcribe",
            generation_config=generation_config,
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " Kimura-san called me."
        unittest.TestCase().assertEqual(transcript, EXPECTED_TRANSCRIPT)

        generated_ids = model.generate(
            input_features,
            do_sample=False,
            max_length=20,
            language="<|ja|>",
            task="translate",
            generation_config=generation_config,
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " I borrowed a phone from Kimura san"
        unittest.TestCase().assertEqual(transcript, EXPECTED_TRANSCRIPT)
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def _test_large_batched_generation(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

        input_speech = _load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features
        generated_ids_1 = model.generate(input_features[0:2], max_length=20)
        generated_ids_2 = model.generate(input_features[2:4], max_length=20)
        generated_ids = np.concatenate([generated_ids_1, generated_ids_2])

        # fmt: off
        EXPECTED_IDS = [
            [50258, 50259, 50359, 50363, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 293, 321, 366, 5404],
            [50258, 50259, 50359, 50363, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50257],
            [50258, 50259, 50359, 50363, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904],
            [50258, 50259, 50359, 50363, 634, 575, 12525, 22618, 1968, 6144, 35617, 20084, 1756, 311, 589, 307, 534, 10281, 934, 439]
        ]
        # fmt: on

        unittest.TestCase().assertEqual(generated_ids.tolist(), EXPECTED_IDS)

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes and we are glad",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all"
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        unittest.TestCase().assertListEqual(transcript, EXPECTED_TRANSCRIPT)
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


@require_tf
@require_tokenizers
class TFWhisperModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return WhisperProcessor.from_pretrained("openai/whisper-base")

    def _load_datasamples(self, num_samples):
        return _load_datasamples(num_samples)

    @slow
    def test_tiny_logits_librispeech(self):
        set_seed(0)
        model = TFWhisperModel.from_pretrained("openai/whisper-tiny")
        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="tf").input_features

        logits = model(
            input_features,
            decoder_input_ids=tf.convert_to_tensor([[50258, 50259, 50359]]),
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
            use_cache=False,
        )

        # fmt: off
        EXPECTED_LOGITS = tf.convert_to_tensor(
            [
                2.9892, -6.7607, 5.7348, 3.6096, 0.2152, -5.7321, 4.8855, -1.6407,
                0.2823, -1.5718, 10.4269, 3.4427, 0.0219, -8.0612, 3.4784, 8.4246,
                4.0575, -2.2864, 11.1084, 0.9963, 0.9884, -8.5154, -3.5469, -9.3713,
                0.9786, 3.5435, 7.4850, -5.2579, -1.4366, 10.4841
            ]
        )
        # fmt: on
        self.assertTrue(np.allclose(logits[0][0, 0, :30], EXPECTED_LOGITS, atol=1e-4))

        # fmt: off
        EXPECTED_GENERATION = tf.convert_to_tensor(
            [
                -1.4651, -2.6944, 2.7821, 2.3793, 4.0738, 0.0188, -3.3203, 1.9836,
                0.0520, 0.7095, 1.1063, 0.2952, -3.6786, -0.5249, 0.3105, 4.7691,
                1.1562, 1.3046, 0.5810, -0.3624, 1.7006, 1.3424, 0.9817, 2.1958,
                1.8775, -5.7046, -0.7679, 4.0113, 2.6848, 2.8609
            ]
        )
        # fmt: on

        head_logits = logits[0] @ tf.transpose(model.model.decoder.embed_tokens.weights[0])
        self.assertTrue(np.allclose(head_logits[0, 0, :30], EXPECTED_GENERATION, atol=1e-4))

    @slow
    def test_small_en_logits_librispeech(self):
        set_seed(0)
        model = TFWhisperModel.from_pretrained("openai/whisper-small.en")

        input_speech = self._load_datasamples(1)

        feaure_extractor = WhisperFeatureExtractor()
        input_features = feaure_extractor(input_speech, return_tensors="tf").input_features

        logits = model(
            input_features,
            decoder_input_ids=tf.convert_to_tensor([[model.config.decoder_start_token_id]]),
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        logits = logits.last_hidden_state @ tf.transpose(model.model.decoder.embed_tokens.weights[0])

        # fmt: off
        EXPECTED_LOGITS = tf.convert_to_tensor(
            [
                -3.6784, -7.7211, -9.5070, -11.9286, -7.6489, -9.7026, -5.6188,
                -8.0104, -4.6238, -5.1833, -9.0485, -3.4079, -5.4874, -2.6935,
                -6.3479, -7.3398, -6.9558, -7.6867, -7.4748, -8.3463, -9.9781,
                -10.8389, -10.3105, -11.7201, -9.7261, -7.1590, -5.9272, -12.4509,
                -11.1146, -8.1918
            ]
        )
        # fmt: on
        self.assertTrue(np.allclose(logits[0, 0, :30], EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_large_logits_librispeech(self):
        run_test_in_subprocess(test_case=self, target_func=_test_large_logits_librispeech, inputs=None)

    @slow
    def test_tiny_en_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model.config.decoder_start_token_id = 50257

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features

        generated_ids = model.generate(input_features, num_beams=5, max_length=20)
        transcript = processor.tokenizer.batch_decode(generated_ids)[0]

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes, and we are glad to"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features

        generated_ids = model.generate(input_features, num_beams=5, max_length=20)
        transcript = processor.tokenizer.decode(generated_ids[0])

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes and we are glad"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_xla_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features

        xla_generate = tf.function(model.generate, jit_compile=True)

        generated_ids = model.generate(input_features, num_beams=5, max_length=20)
        generated_ids_xla = xla_generate(input_features, num_beams=5, max_length=20)

        transcript = processor.tokenizer.decode(generated_ids[0])
        transcript_xla = processor.tokenizer.decode(generated_ids_xla[0])

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes and we are glad"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)
        self.assertEqual(transcript_xla, EXPECTED_TRANSCRIPT)

    @slow
    def test_large_generation(self):
        run_test_in_subprocess(test_case=self, target_func=_test_large_generation, inputs=None)

    @slow
    @require_read_token
    def test_large_generation_multilingual(self):
        run_test_in_subprocess(test_case=self, target_func=_test_large_generation_multilingual, inputs=None)

    @slow
    def test_large_batched_generation(self):
        run_test_in_subprocess(test_case=self, target_func=_test_large_batched_generation, inputs=None)

    @slow
    def test_tiny_en_batched_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features
        generated_ids = model.generate(input_features, max_length=20)

        # fmt: off
        EXPECTED_LOGITS = tf.convert_to_tensor(
            [
                [50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284],
                [50257, 50362, 5414, 318, 1770, 13, 2264, 346, 353, 338, 5642, 1342, 3499, 621, 465, 2300, 13, 50256, 50256, 50256],
                [50257, 50362, 679, 4952, 514, 326, 379, 428, 43856, 1622, 286, 262, 614, 11, 351, 6786, 290, 32595, 12023, 28236],
                [50257, 50362, 679, 468, 12296, 17188, 1771, 7361, 26113, 18881, 1122, 338, 670, 318, 1107, 8312, 706, 477, 290, 460]
            ]

        )
        # fmt: on

        self.assertTrue(np.allclose(generated_ids, EXPECTED_LOGITS))

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes, and we are glad to",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast beef looming",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can",
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_en_batched_xla_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="tf").input_features

        xla_generate = tf.function(model.generate, jit_compile=True)

        generated_ids = model.generate(input_features, max_length=20)
        generated_ids_xla = xla_generate(input_features, max_length=20)

        # fmt: off
        EXPECTED_LOGITS = tf.convert_to_tensor(
            [
                [50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284],
                [50257, 50362, 5414, 318, 1770, 13, 2264, 346, 353, 338, 5642, 1342, 3499, 621, 465, 2300, 13, 50256, 50256, 50256],
                [50257, 50362, 679, 4952, 514, 326, 379, 428, 43856, 1622, 286, 262, 614, 11, 351, 6786, 290, 32595, 12023, 28236],
                [50257, 50362, 679, 468, 12296, 17188, 1771, 7361, 26113, 18881, 1122, 338, 670, 318, 1107, 8312, 706, 477, 290, 460]
            ]

        )
        # fmt: on

        self.assertTrue(np.allclose(generated_ids, EXPECTED_LOGITS))
        self.assertTrue(np.allclose(generated_ids_xla, EXPECTED_LOGITS))

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes, and we are glad to",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast beef looming",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can",
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        transcript_xla = processor.batch_decode(generated_ids_xla, skip_special_tokens=True)
        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)
        self.assertListEqual(transcript_xla, EXPECTED_TRANSCRIPT)
