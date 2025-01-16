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
"""Testing suite for the PyTorch Informer model."""

import inspect
import tempfile
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import is_torch_available
from transformers.testing_utils import is_flaky, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


TOLERANCE = 1e-4

if is_torch_available():
    import torch

    from transformers import InformerConfig, InformerForPrediction, InformerModel
    from transformers.models.informer.modeling_informer import (
        InformerDecoder,
        InformerEncoder,
        InformerSinusoidalPositionalEmbedding,
    )


@require_torch
class InformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        prediction_length=7,
        context_length=14,
        cardinality=19,
        embedding_dimension=5,
        num_time_features=4,
        is_training=True,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        lags_sequence=[1, 2, 3, 4, 5],
        sampling_factor=10,
        distil=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.cardinality = cardinality
        self.num_time_features = num_time_features
        self.lags_sequence = lags_sequence
        self.embedding_dimension = embedding_dimension
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.encoder_seq_length = min(
            sampling_factor * np.ceil(np.log1p(context_length)).astype("int").item(), context_length
        )
        self.decoder_seq_length = min(
            sampling_factor * np.ceil(np.log1p(prediction_length)).astype("int").item(), prediction_length
        )
        self.sampling_factor = sampling_factor
        self.distil = distil

    def get_config(self):
        return InformerConfig(
            prediction_length=self.prediction_length,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            context_length=self.context_length,
            lags_sequence=self.lags_sequence,
            num_time_features=self.num_time_features,
            num_static_categorical_features=1,
            num_static_real_features=1,
            cardinality=[self.cardinality],
            embedding_dimension=[self.embedding_dimension],
            sampling_factor=self.sampling_factor,
            distil=self.distil,
        )

    def prepare_informer_inputs_dict(self, config):
        _past_length = config.context_length + max(config.lags_sequence)

        static_categorical_features = ids_tensor([self.batch_size, 1], config.cardinality[0])
        static_real_features = floats_tensor([self.batch_size, 1])

        past_time_features = floats_tensor([self.batch_size, _past_length, config.num_time_features])
        past_values = floats_tensor([self.batch_size, _past_length])
        past_observed_mask = floats_tensor([self.batch_size, _past_length]) > 0.5

        # decoder inputs
        future_time_features = floats_tensor([self.batch_size, config.prediction_length, config.num_time_features])
        future_values = floats_tensor([self.batch_size, config.prediction_length])

        inputs_dict = {
            "past_values": past_values,
            "static_categorical_features": static_categorical_features,
            "static_real_features": static_real_features,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "future_time_features": future_time_features,
            "future_values": future_values,
        }
        return inputs_dict

    def prepare_config_and_inputs(self):
        config = self.get_config()
        inputs_dict = self.prepare_informer_inputs_dict(config)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = InformerModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = InformerEncoder.from_pretrained(tmpdirname).to(torch_device)

        transformer_inputs, _, _, _ = model.create_network_inputs(**inputs_dict)
        enc_input = transformer_inputs[:, : config.context_length, ...]
        dec_input = transformer_inputs[:, config.context_length :, ...]

        encoder_last_hidden_state_2 = encoder(inputs_embeds=enc_input)[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        embed_positions = InformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        ).to(torch_device)
        self.parent.assertTrue(torch.equal(model.encoder.embed_positions.weight, embed_positions.weight))
        self.parent.assertTrue(torch.equal(model.decoder.embed_positions.weight, embed_positions.weight))

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = InformerDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            inputs_embeds=dec_input,
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class InformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (InformerModel, InformerForPrediction) if is_torch_available() else ()
    all_generative_model_classes = (InformerForPrediction,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": InformerModel} if is_torch_available() else {}
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = InformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=InformerConfig,
            has_text_modality=False,
            prediction_length=self.model_tester.prediction_length,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

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

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.context_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "prediction_length", seq_len)

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

    @unittest.skip(reason="Informer does not have tokens embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip
    def test_determinism(self):
        pass

    @unittest.skip(reason="randomly selects U keys while calculating attentions")
    def test_batching_equivalence(self):
        pass

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

    # # Input is 'static_categorical_features' not 'input_ids'
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(InformerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(InformerModel.main_input_name, observed_main_input_name)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "past_values",
                "past_time_features",
                "past_observed_mask",
                "static_categorical_features",
                "static_real_features",
                "future_values",
                "future_time_features",
            ]

            expected_arg_names.extend(
                [
                    "future_observed_mask",
                    "decoder_attention_mask",
                    "head_mask",
                    "decoder_head_mask",
                    "cross_attn_head_mask",
                    "encoder_outputs",
                    "past_key_values",
                    "output_hidden_states",
                    "output_attentions",
                    "use_cache",
                    "return_dict",
                ]
                if "future_observed_mask" in arg_names
                else [
                    "decoder_attention_mask",
                    "head_mask",
                    "decoder_head_mask",
                    "cross_attn_head_mask",
                    "encoder_outputs",
                    "past_key_values",
                    "output_hidden_states",
                    "output_attentions",
                    "use_cache",
                    "return_dict",
                ]
            )

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        context_length = getattr(self.model_tester, "context_length", seq_len)
        prediction_length = getattr(self.model_tester, "prediction_length", seq_len)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, context_length],
            )
            out_len = len(outputs)

            correct_outlen = 7

            if "last_hidden_state" in outputs:
                correct_outlen += 1

            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            if "loss" in outputs:
                correct_outlen += 1

            if "params" in outputs:
                correct_outlen += 1

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, prediction_length],
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
                    encoder_seq_length,
                ],
            )

        # Check attention is always last and order is fine
        inputs_dict["output_attentions"] = True
        inputs_dict["output_hidden_states"] = True
        model = model_class(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

        self.assertEqual(out_len + 2, len(outputs))

        self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

        self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
        self.assertListEqual(
            list(self_attentions[0].shape[-3:]),
            [self.model_tester.num_attention_heads, encoder_seq_length, context_length],
        )

    @is_flaky()
    def test_retain_grad_hidden_states_attentions(self):
        super().test_retain_grad_hidden_states_attentions()

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass


def prepare_batch(filename="train-batch.pt"):
    file = hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename=filename, repo_type="dataset")
    batch = torch.load(file, map_location=torch_device)
    return batch


@require_torch
@slow
class InformerModelIntegrationTests(unittest.TestCase):
    def test_inference_no_head(self):
        model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly").to(torch_device)
        batch = prepare_batch()

        torch.manual_seed(0)
        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            ).last_hidden_state
        expected_shape = torch.Size((64, model.config.context_length, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.4699, 0.7295, 0.8967], [0.4858, 0.3810, 0.9641], [-0.0233, 0.3608, 1.0303]],
            device=torch_device,
        )
        torch.testing.assert_close(output[0, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    def test_inference_head(self):
        model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly").to(torch_device)
        batch = prepare_batch("val-batch.pt")

        torch.manual_seed(0)
        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                future_time_features=batch["future_time_features"],
            ).encoder_last_hidden_state

        # encoder distils the context length to 1/8th of the original length
        expected_shape = torch.Size((64, model.config.context_length // 8, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.4170, 0.9067, 0.8153], [0.3004, 0.7574, 0.7066], [0.6803, -0.6323, 1.2802]], device=torch_device
        )
        torch.testing.assert_close(output[0, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    def test_seq_to_seq_generation(self):
        model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly").to(torch_device)
        batch = prepare_batch("val-batch.pt")

        torch.manual_seed(0)
        with torch.no_grad():
            outputs = model.generate(
                static_categorical_features=batch["static_categorical_features"],
                past_time_features=batch["past_time_features"],
                past_values=batch["past_values"],
                future_time_features=batch["future_time_features"],
                past_observed_mask=batch["past_observed_mask"],
            )
        expected_shape = torch.Size((64, model.config.num_parallel_samples, model.config.prediction_length))
        self.assertEqual(outputs.sequences.shape, expected_shape)

        expected_slice = torch.tensor([3400.8005, 4289.2637, 7101.9209], device=torch_device)
        mean_prediction = outputs.sequences.mean(dim=1)
        torch.testing.assert_close(mean_prediction[0, -3:], expected_slice, rtol=1e-1)
