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
"""Testing suite for the PyTorch TimeSeriesTransformer model."""

import inspect
import tempfile
import unittest

from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import is_flaky, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


TOLERANCE = 1e-4

if is_torch_available():
    import torch

    from transformers import (
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerForPrediction,
        TimeSeriesTransformerModel,
    )
    from transformers.models.time_series_transformer.modeling_time_series_transformer import (
        TimeSeriesTransformerDecoder,
        TimeSeriesTransformerEncoder,
    )


@require_torch
class TimeSeriesTransformerModelTester:
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
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        lags_sequence=[1, 2, 3, 4, 5],
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

        self.encoder_seq_length = context_length
        self.decoder_seq_length = prediction_length

    def get_config(self):
        return TimeSeriesTransformerConfig(
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            lags_sequence=self.lags_sequence,
            num_time_features=self.num_time_features,
            num_static_real_features=1,
            num_static_categorical_features=1,
            cardinality=[self.cardinality],
            embedding_dimension=[self.embedding_dimension],
            scaling="std",  # we need std to get non-zero `loc`
        )

    def prepare_time_series_transformer_inputs_dict(self, config):
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
        inputs_dict = self.prepare_time_series_transformer_inputs_dict(config)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = TimeSeriesTransformerModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = TimeSeriesTransformerEncoder.from_pretrained(tmpdirname).to(torch_device)

        transformer_inputs, _, _, _ = model.create_network_inputs(**inputs_dict)
        enc_input = transformer_inputs[:, : config.context_length, ...]
        dec_input = transformer_inputs[:, config.context_length :, ...]

        encoder_last_hidden_state_2 = encoder(inputs_embeds=enc_input)[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = TimeSeriesTransformerDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            inputs_embeds=dec_input,
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class TimeSeriesTransformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction) if is_torch_available() else ()
    )
    all_generative_model_classes = (TimeSeriesTransformerForPrediction,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": TimeSeriesTransformerModel} if is_torch_available() else {}
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = TimeSeriesTransformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=TimeSeriesTransformerConfig,
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

    @unittest.skip(reason="Model has no tokens embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    # # Input is 'static_categorical_features' not 'input_ids'
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(TimeSeriesTransformerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(TimeSeriesTransformerModel.main_input_name, observed_main_input_name)

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
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_seq_length],
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
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_seq_length],
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
            [self.model_tester.num_attention_heads, encoder_seq_length, encoder_seq_length],
        )

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

    @parameterized.expand(
        [
            (1, 5, [1]),
            (1, 5, [1, 10, 15]),
            (1, 5, [3, 6, 9, 10]),
            (2, 5, [1, 2, 7]),
            (2, 5, [2, 3, 4, 6]),
            (4, 5, [1, 5, 9, 11]),
            (4, 5, [7, 8, 13, 14]),
        ],
    )
    def test_create_network_inputs(self, prediction_length, context_length, lags_sequence):
        history_length = max(lags_sequence) + context_length

        config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            lags_sequence=lags_sequence,
            scaling=False,
            num_parallel_samples=10,
            num_static_categorical_features=1,
            cardinality=[1],
            embedding_dimension=[2],
            num_static_real_features=1,
        )
        model = TimeSeriesTransformerModel(config)

        batch = {
            "static_categorical_features": torch.tensor([[0]], dtype=torch.int64),
            "static_real_features": torch.tensor([[0.0]], dtype=torch.float32),
            "past_time_features": torch.arange(history_length, dtype=torch.float32).view(1, history_length, 1),
            "past_values": torch.arange(history_length, dtype=torch.float32).view(1, history_length),
            "past_observed_mask": torch.arange(history_length, dtype=torch.float32).view(1, history_length),
        }

        # test with no future_target (only one step prediction)
        batch["future_time_features"] = torch.arange(history_length, history_length + 1, dtype=torch.float32).view(
            1, 1, 1
        )
        transformer_inputs, loc, scale, _ = model.create_network_inputs(**batch)

        self.assertTrue((scale == 1.0).all())
        assert (loc == 0.0).all()

        ref = torch.arange(max(lags_sequence), history_length, dtype=torch.float32)

        for idx, lag in enumerate(lags_sequence):
            assert torch.isclose(ref - lag, transformer_inputs[0, :, idx]).all()

        # test with all future data
        batch["future_time_features"] = torch.arange(
            history_length, history_length + prediction_length, dtype=torch.float32
        ).view(1, prediction_length, 1)
        batch["future_values"] = torch.arange(
            history_length, history_length + prediction_length, dtype=torch.float32
        ).view(1, prediction_length)
        transformer_inputs, loc, scale, _ = model.create_network_inputs(**batch)

        assert (scale == 1.0).all()
        assert (loc == 0.0).all()

        ref = torch.arange(max(lags_sequence), history_length + prediction_length, dtype=torch.float32)

        for idx, lag in enumerate(lags_sequence):
            assert torch.isclose(ref - lag, transformer_inputs[0, :, idx]).all()

        # test for generation
        batch.pop("future_values")
        transformer_inputs, loc, scale, _ = model.create_network_inputs(**batch)

        lagged_sequence = model.get_lagged_subsequences(
            sequence=batch["past_values"],
            subsequences_length=1,
            shift=1,
        )
        # assert that the last element of the lagged sequence is the one after the encoders input
        assert transformer_inputs[0, ..., 0][-1] + 1 == lagged_sequence[0, ..., 0][-1]

        future_values = torch.arange(history_length, history_length + prediction_length, dtype=torch.float32).view(
            1, prediction_length
        )
        # assert that the first element of the future_values is offset by lag after the decoders input
        assert lagged_sequence[0, ..., 0][-1] + lags_sequence[0] == future_values[0, ..., 0]

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
class TimeSeriesTransformerModelIntegrationTests(unittest.TestCase):
    def test_inference_no_head(self):
        model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly").to(
            torch_device
        )
        batch = prepare_batch()

        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                static_real_features=batch["static_real_features"],
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            ).last_hidden_state

        expected_shape = torch.Size((64, model.config.context_length, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.8196, -1.5131, 1.4620], [1.1268, -1.3238, 1.5997], [1.5098, -1.0715, 1.7359]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_head(self):
        model = TimeSeriesTransformerForPrediction.from_pretrained(
            "huggingface/time-series-transformer-tourism-monthly"
        ).to(torch_device)
        batch = prepare_batch("val-batch.pt")
        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                static_real_features=batch["static_real_features"],
                future_time_features=batch["future_time_features"],
            ).encoder_last_hidden_state
        expected_shape = torch.Size((64, model.config.context_length, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-1.2957, -1.0280, -0.6045], [-0.7017, -0.8193, -0.3717], [-1.0449, -0.8149, 0.1405]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        model = TimeSeriesTransformerForPrediction.from_pretrained(
            "huggingface/time-series-transformer-tourism-monthly"
        ).to(torch_device)
        batch = prepare_batch("val-batch.pt")
        with torch.no_grad():
            outputs = model.generate(
                static_categorical_features=batch["static_categorical_features"],
                static_real_features=batch["static_real_features"],
                past_time_features=batch["past_time_features"],
                past_values=batch["past_values"],
                future_time_features=batch["future_time_features"],
                past_observed_mask=batch["past_observed_mask"],
            )
        expected_shape = torch.Size((64, model.config.num_parallel_samples, model.config.prediction_length))
        self.assertEqual(outputs.sequences.shape, expected_shape)
        self.assertEqual(outputs.distribution, "student_t")
        self.assertEqual(outputs.params["df"].snape, expected_shape)
        self.assertEqual(outputs.params["loc"].snape, expected_shape)
        self.assertEqual(outputs.params["scale"].snape, expected_shape)
        self.assertEqual(len(outputs.scaling_params), 2)

        expected_slice = torch.tensor([2825.2749, 3584.9207, 6763.9951], device=torch_device)
        mean_prediction = outputs.sequences.mean(dim=1)
        self.assertTrue(torch.allclose(mean_prediction[0, -3:], expected_slice, rtol=1e-1))
