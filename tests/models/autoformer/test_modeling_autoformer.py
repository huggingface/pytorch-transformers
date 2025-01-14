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
"""Testing suite for the PyTorch Autoformer model."""

import inspect
import tempfile
import unittest

from huggingface_hub import hf_hub_download

from transformers import is_torch_available
from transformers.testing_utils import is_flaky, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


TOLERANCE = 1e-4

if is_torch_available():
    import torch

    from transformers import AutoformerConfig, AutoformerForPrediction, AutoformerModel
    from transformers.models.autoformer.modeling_autoformer import AutoformerDecoder, AutoformerEncoder


@require_torch
class AutoformerModelTester:
    def __init__(
        self,
        parent,
        d_model=16,
        batch_size=13,
        prediction_length=7,
        context_length=14,
        label_length=10,
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
        moving_average=25,
        autocorrelation_factor=5,
    ):
        self.d_model = d_model
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
        self.decoder_seq_length = prediction_length + label_length
        self.label_length = label_length

        self.moving_average = moving_average
        self.autocorrelation_factor = autocorrelation_factor

    def get_config(self):
        return AutoformerConfig(
            d_model=self.d_model,
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
            label_length=self.label_length,
            lags_sequence=self.lags_sequence,
            num_time_features=self.num_time_features,
            num_static_categorical_features=1,
            cardinality=[self.cardinality],
            embedding_dimension=[self.embedding_dimension],
            moving_average=self.moving_average,
            scaling="std",  # we need std to get non-zero `loc`
        )

    def prepare_autoformer_inputs_dict(self, config):
        _past_length = config.context_length + max(config.lags_sequence)

        static_categorical_features = ids_tensor([self.batch_size, 1], config.cardinality[0])
        past_time_features = floats_tensor([self.batch_size, _past_length, config.num_time_features])
        past_values = floats_tensor([self.batch_size, _past_length])
        past_observed_mask = floats_tensor([self.batch_size, _past_length]) > 0.5

        # decoder inputs
        future_time_features = floats_tensor([self.batch_size, config.prediction_length, config.num_time_features])
        future_values = floats_tensor([self.batch_size, config.prediction_length])

        inputs_dict = {
            "past_values": past_values,
            "static_categorical_features": static_categorical_features,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "future_time_features": future_time_features,
            "future_values": future_values,
        }
        return inputs_dict

    def prepare_config_and_inputs(self):
        config = self.get_config()
        inputs_dict = self.prepare_autoformer_inputs_dict(config)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = AutoformerModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = AutoformerEncoder.from_pretrained(tmpdirname).to(torch_device)

        transformer_inputs, feature, _, _, _ = model.create_network_inputs(**inputs_dict)
        seasonal_input, trend_input = model.decomposition_layer(transformer_inputs[:, : config.context_length, ...])

        enc_input = torch.cat(
            (transformer_inputs[:, : config.context_length, ...], feature[:, : config.context_length, ...]),
            dim=-1,
        )
        encoder_last_hidden_state_2 = encoder(inputs_embeds=enc_input)[0]
        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        mean = (
            torch.mean(transformer_inputs[:, : config.context_length, ...], dim=1)
            .unsqueeze(1)
            .repeat(1, config.prediction_length, 1)
        )
        zeros = torch.zeros(
            [transformer_inputs.shape[0], config.prediction_length, transformer_inputs.shape[2]],
            device=enc_input.device,
        )

        dec_input = torch.cat(
            (
                torch.cat((seasonal_input[:, -config.label_length :, ...], zeros), dim=1),
                feature[:, config.context_length - config.label_length :, ...],
            ),
            dim=-1,
        )
        trend_init = torch.cat(
            (
                torch.cat((trend_input[:, -config.label_length :, ...], mean), dim=1),
                feature[:, config.context_length - config.label_length :, ...],
            ),
            dim=-1,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = AutoformerDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            trend=trend_init,
            inputs_embeds=dec_input,
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class AutoformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (AutoformerModel, AutoformerForPrediction) if is_torch_available() else ()
    all_generative_model_classes = (AutoformerForPrediction,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": AutoformerModel} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = AutoformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AutoformerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
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

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    # # Input is 'static_categorical_features' not 'input_ids'
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(AutoformerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(AutoformerModel.main_input_name, observed_main_input_name)

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

            if model.__class__.__name__ in ["AutoformerForPrediction"]:
                expected_arg_names.append("future_observed_mask")

            expected_arg_names.extend(
                [
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
        d_model = getattr(self.model_tester, "d_model", None)
        num_attention_heads = getattr(self.model_tester, "num_attention_heads", None)
        dim = d_model // num_attention_heads

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
                [self.model_tester.num_attention_heads, encoder_seq_length, dim],
            )
            out_len = len(outputs)

            correct_outlen = 7

            if "last_hidden_state" in outputs:
                correct_outlen += 1

            if "trend" in outputs:
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
                [self.model_tester.num_attention_heads, decoder_seq_length, dim],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, dim],
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
            [self.model_tester.num_attention_heads, encoder_seq_length, dim],
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
class AutoformerModelIntegrationTests(unittest.TestCase):
    def test_inference_no_head(self):
        model = AutoformerModel.from_pretrained("huggingface/autoformer-tourism-monthly").to(torch_device)
        batch = prepare_batch()

        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            )[0]

        expected_shape = torch.Size(
            (64, model.config.prediction_length + model.config.label_length, model.config.feature_size)
        )
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.3593, -1.3398, 0.6330], [0.2279, 1.5396, -0.1792], [0.0450, 1.3225, -0.2335]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_head(self):
        model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly").to(torch_device)
        batch = prepare_batch("val-batch.pt")
        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
            ).encoder_last_hidden_state
        expected_shape = torch.Size((64, model.config.context_length, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.0734, -0.9036, 0.8358], [4.7186, 2.4113, 1.9581], [1.7953, 2.3558, 1.2970]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly").to(torch_device)
        batch = prepare_batch("val-batch.pt")
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
        self.assertEqual(outputs.distribution, "student_t")
        self.assertEqual(outputs.params["df"].snape, expected_shape)
        self.assertEqual(outputs.params["loc"].snape, expected_shape)
        self.assertEqual(outputs.params["scale"].snape, expected_shape)
        self.assertEqual(len(outputs.scaling_params), 2)

        expected_slice = torch.tensor([3130.6763, 4056.5293, 7053.0786], device=torch_device)
        mean_prediction = outputs.sequences.mean(dim=1)
        self.assertTrue(torch.allclose(mean_prediction[0, -3:], expected_slice, rtol=1e-1))
