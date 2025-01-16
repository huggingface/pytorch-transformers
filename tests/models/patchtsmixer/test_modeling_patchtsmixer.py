# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch PatchTSMixer model."""

import inspect
import itertools
import random
import tempfile
import unittest
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import is_flaky, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


TOLERANCE = 1e-4

if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
        MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
        PatchTSMixerConfig,
        PatchTSMixerForPrediction,
        PatchTSMixerForPretraining,
        PatchTSMixerForRegression,
        PatchTSMixerForTimeSeriesClassification,
        PatchTSMixerModel,
    )
    from transformers.models.patchtsmixer.modeling_patchtsmixer import (
        PatchTSMixerEncoder,
        PatchTSMixerForPredictionHead,
        PatchTSMixerForPredictionOutput,
        PatchTSMixerForRegressionOutput,
        PatchTSMixerForTimeSeriesClassificationOutput,
        PatchTSMixerLinearHead,
        PatchTSMixerPretrainHead,
    )


@require_torch
class PatchTSMixerModelTester:
    def __init__(
        self,
        context_length: int = 32,
        patch_length: int = 8,
        num_input_channels: int = 3,
        patch_stride: int = 8,
        # d_model: int = 128,
        hidden_size: int = 8,
        # num_layers: int = 8,
        num_hidden_layers: int = 2,
        expansion_factor: int = 2,
        dropout: float = 0.5,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp="LayerNorm",
        swin_hier: int = 0,
        # masking related
        mask_type: str = "forecast",
        random_mask_ratio=0.5,
        mask_patches: list = [2, 3],
        forecast_mask_ratios: list = [1, 1],
        mask_value=0,
        masked_loss: bool = False,
        mask_mode: str = "mask_before_encoder",
        channel_consistent_masking: bool = True,
        scaling: Optional[Union[str, bool]] = "std",
        # Head related
        head_dropout: float = 0.2,
        # forecast related
        prediction_length: int = 16,
        out_channels: int = None,
        # Classification/regression related
        # num_labels: int = 3,
        num_targets: int = 3,
        output_range: list = None,
        head_aggregation: str = None,
        # Trainer related
        batch_size=13,
        is_training=True,
        seed_number=42,
        post_init=True,
        num_parallel_samples=4,
    ):
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        # self.d_model = d_model
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        # self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.swin_hier = swin_hier
        self.scaling = scaling
        self.head_dropout = head_dropout
        # masking related
        self.mask_type = mask_type
        self.random_mask_ratio = random_mask_ratio
        self.mask_patches = mask_patches
        self.forecast_mask_ratios = forecast_mask_ratios
        self.mask_value = mask_value
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_mode = mask_mode
        self.masked_loss = masked_loss
        # patching related
        self.patch_last = True
        # forecast related
        self.prediction_length = prediction_length
        self.out_channels = out_channels
        # classification/regression related
        # self.num_labels = num_labels
        self.num_targets = num_targets
        self.output_range = output_range
        self.head_aggregation = head_aggregation
        # Trainer related
        self.batch_size = batch_size
        self.is_training = is_training
        self.seed_number = seed_number
        self.post_init = post_init
        self.num_parallel_samples = num_parallel_samples

    def get_config(self):
        config_ = PatchTSMixerConfig(
            num_input_channels=self.num_input_channels,
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            # d_model = self.d_model,
            d_model=self.hidden_size,
            expansion_factor=self.expansion_factor,
            # num_layers = self.num_layers,
            num_layers=self.num_hidden_layers,
            dropout=self.dropout,
            mode=self.mode,
            gated_attn=self.gated_attn,
            norm_mlp=self.norm_mlp,
            swin_hier=self.swin_hier,
            scaling=self.scaling,
            head_dropout=self.head_dropout,
            mask_type=self.mask_type,
            random_mask_ratio=self.random_mask_ratio,
            mask_patches=self.mask_patches,
            forecast_mask_ratios=self.forecast_mask_ratios,
            mask_value=self.mask_value,
            channel_consistent_masking=self.channel_consistent_masking,
            mask_mode=self.mask_mode,
            masked_loss=self.masked_loss,
            prediction_length=self.prediction_length,
            out_channels=self.out_channels,
            # num_labels=self.num_labels,
            num_targets=self.num_targets,
            output_range=self.output_range,
            head_aggregation=self.head_aggregation,
            post_init=self.post_init,
        )
        self.num_patches = config_.num_patches
        return config_

    def prepare_patchtsmixer_inputs_dict(self, config):
        _past_length = config.context_length
        # bs, n_vars, num_patch, patch_length

        # [bs x context_length x n_vars]
        past_values = floats_tensor([self.batch_size, _past_length, self.num_input_channels])

        inputs_dict = {
            "past_values": past_values,
        }
        return inputs_dict

    def prepare_config_and_inputs(self):
        config = self.get_config()
        inputs_dict = self.prepare_patchtsmixer_inputs_dict(config)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class PatchTSMixerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            PatchTSMixerModel,
            PatchTSMixerForPrediction,
            PatchTSMixerForPretraining,
            PatchTSMixerForTimeSeriesClassification,
            PatchTSMixerForRegression,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (
        (PatchTSMixerForPrediction, PatchTSMixerForPretraining) if is_torch_available() else ()
    )
    pipeline_model_mapping = {"feature-extraction": PatchTSMixerModel} if is_torch_available() else {}
    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False
    test_inputs_embeds = False

    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_model_parallel = False
    has_attentions = False

    def setUp(self):
        self.model_tester = PatchTSMixerModelTester()
        self.config_tester = ConfigTester(
            self,
            config_class=PatchTSMixerConfig,
            has_text_modality=False,
            prediction_length=self.model_tester.prediction_length,
            common_properties=["hidden_size", "expansion_factor", "num_hidden_layers"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if model_class == PatchTSMixerForPrediction:
            rng = random.Random(self.model_tester.seed_number)
            labels = floats_tensor(
                [
                    self.model_tester.batch_size,
                    self.model_tester.prediction_length,
                    self.model_tester.num_input_channels,
                ],
                rng=rng,
            )
            inputs_dict["future_values"] = labels
        elif model_class in get_values(MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING):
            rng = random.Random(self.model_tester.seed_number)
            labels = ids_tensor([self.model_tester.batch_size], self.model_tester.num_targets, rng=rng)
            inputs_dict["target_values"] = labels
        elif model_class in get_values(MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING):
            rng = random.Random(self.model_tester.seed_number)
            labels = floats_tensor([self.model_tester.batch_size, self.model_tester.num_targets], rng=rng)
            inputs_dict["target_values"] = labels

        inputs_dict["output_hidden_states"] = True
        return inputs_dict

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester,
                "expected_num_hidden_layers",
                self.model_tester.num_hidden_layers,
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            expected_hidden_size = self.model_tester.hidden_size
            self.assertEqual(hidden_states[0].shape[-1], expected_hidden_size)

            num_patch = self.model_tester.num_patches
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [num_patch, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="No tokens embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                output_ = model(**dict_inputs, return_dict=True, **additional_kwargs)
                attributes_ = vars(output_)
                dict_output = tuple(attributes_.values())

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object),
                                set_nan_tensor_to_zero(dict_object),
                                atol=1e-5,
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            print(model_class)
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)

            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            tuple_inputs.update({"output_hidden_states": False})
            dict_inputs.update({"output_hidden_states": False})
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            tuple_inputs.update({"output_hidden_states": False})
            dict_inputs.update({"output_hidden_states": False})
            check_equivalence(
                model,
                tuple_inputs,
                dict_inputs,
            )

    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(PatchTSMixerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(PatchTSMixerModel.main_input_name, observed_main_input_name)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model_class == PatchTSMixerForPretraining:
                expected_arg_names = [
                    "past_values",
                    "observed_mask",
                    "output_hidden_states",
                    "return_loss",
                ]
            elif model_class == PatchTSMixerModel:
                expected_arg_names = [
                    "past_values",
                    "observed_mask",
                    "output_hidden_states",
                ]
            elif model_class in get_values(MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING) or model_class in get_values(
                MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING
            ):
                expected_arg_names = [
                    "past_values",
                    "target_values",
                    "output_hidden_states",
                    "return_loss",
                ]
            else:
                # PatchTSMixerForPrediction
                expected_arg_names = [
                    "past_values",
                    "observed_mask",
                    "future_values",
                    "output_hidden_states",
                    "return_loss",
                ]

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @is_flaky()
    def test_retain_grad_hidden_states_attentions(self):
        super().test_retain_grad_hidden_states_attentions()

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass


def prepare_batch(repo_id="ibm/patchtsmixer-etth1-test-data", file="pretrain_batch.pt"):
    # TODO: Make repo public
    file = hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset")
    batch = torch.load(file, map_location=torch_device)
    return batch


@require_torch
@slow
class PatchTSMixerModelIntegrationTests(unittest.TestCase):
    def test_pretrain_head(self):
        model = PatchTSMixerForPretraining.from_pretrained("ibm/patchtsmixer-etth1-pretrain").to(torch_device)
        batch = prepare_batch()

        torch.manual_seed(0)
        with torch.no_grad():
            output = model(past_values=batch["past_values"].to(torch_device)).prediction_outputs
        num_patch = (
            max(model.config.context_length, model.config.patch_length) - model.config.patch_length
        ) // model.config.patch_stride + 1
        expected_shape = torch.Size(
            [
                64,
                model.config.num_input_channels,
                num_patch,
                model.config.patch_length,
            ]
        )
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[[[-0.9106]],[[1.5326]],[[-0.8245]],[[0.7439]],[[-0.7830]],[[2.6256]],[[-0.6485]],]],device=torch_device)  # fmt: skip
        torch.testing.assert_close(output[0, :7, :1, :1], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    def test_forecasting_head(self):
        model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-etth1-forecasting").to(torch_device)
        batch = prepare_batch(file="forecast_batch.pt")

        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            output = model(
                past_values=batch["past_values"].to(torch_device),
                future_values=batch["future_values"].to(torch_device),
            ).prediction_outputs

        expected_shape = torch.Size([64, model.config.prediction_length, model.config.num_input_channels])
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.2471, 0.5036, 0.3596, 0.5401, -0.0985, 0.3423, -0.8439]],
            device=torch_device,
        )
        torch.testing.assert_close(output[0, :1, :7], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    def test_prediction_generation(self):
        model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-etth1-generate").to(torch_device)
        batch = prepare_batch(file="forecast_batch.pt")
        print(batch["past_values"])

        torch.manual_seed(0)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(past_values=batch["past_values"].to(torch_device))
        expected_shape = torch.Size((64, 1, model.config.prediction_length, model.config.num_input_channels))

        self.assertEqual(outputs.sequences.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.4308, -0.4731, 1.3512, -0.1038, -0.4655, 1.1279, -0.7179]],
            device=torch_device,
        )

        mean_prediction = outputs.sequences.mean(dim=1)

        torch.testing.assert_close(mean_prediction[0, -1:], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)


@require_torch
class PatchTSMixerFunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.params = {}
        cls.params.update(
            context_length=32,
            patch_length=8,
            num_input_channels=3,
            patch_stride=8,
            d_model=4,
            expansion_factor=2,
            num_layers=3,
            dropout=0.2,
            mode="common_channel",  # common_channel,  mix_channel
            gated_attn=True,
            norm_mlp="LayerNorm",
            mask_type="random",
            random_mask_ratio=0.5,
            mask_patches=[2, 3],
            forecast_mask_ratios=[1, 1],
            mask_value=0,
            masked_loss=True,
            channel_consistent_masking=True,
            head_dropout=0.2,
            prediction_length=64,
            out_channels=None,
            # num_labels=3,
            num_targets=3,
            output_range=None,
            head_aggregation=None,
            scaling="std",
            use_positional_encoding=False,
            positional_encoding="sincos",
            self_attn=False,
            self_attn_heads=1,
            num_parallel_samples=4,
        )

        cls.num_patches = (
            max(cls.params["context_length"], cls.params["patch_length"]) - cls.params["patch_length"]
        ) // cls.params["patch_stride"] + 1

        # batch_size = 32
        batch_size = 2

        int(cls.params["prediction_length"] / cls.params["patch_length"])

        cls.data = torch.rand(
            batch_size,
            cls.params["context_length"],
            cls.params["num_input_channels"],
        )

        cls.enc_data = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["patch_length"],
        )

        cls.enc_output = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["d_model"],
        )

        cls.flat_enc_output = torch.rand(
            batch_size,
            cls.num_patches,
            cls.params["d_model"],
        )

        cls.correct_pred_output = torch.rand(
            batch_size,
            cls.params["prediction_length"],
            cls.params["num_input_channels"],
        )
        cls.correct_regression_output = torch.rand(batch_size, cls.params["num_targets"])

        cls.correct_pretrain_output = torch.rand(
            batch_size,
            cls.params["num_input_channels"],
            cls.num_patches,
            cls.params["patch_length"],
        )

        cls.correct_forecast_output = torch.rand(
            batch_size,
            cls.params["prediction_length"],
            cls.params["num_input_channels"],
        )

        cls.correct_sel_forecast_output = torch.rand(batch_size, cls.params["prediction_length"], 2)

        cls.correct_classification_output = torch.rand(
            batch_size,
            cls.params["num_targets"],
        )

        cls.correct_classification_classes = torch.randint(0, cls.params["num_targets"], (batch_size,))

    def test_patchtsmixer_encoder(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        enc = PatchTSMixerEncoder(config)
        output = enc(self.__class__.enc_data)
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)

    def test_patchmodel(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerModel(config)
        output = mdl(self.__class__.data)
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.patch_input.shape, self.__class__.enc_data.shape)

    def test_pretrainhead(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerPretrainHead(
            config=config,
        )
        output = head(self.__class__.enc_output)

        self.assertEqual(output.shape, self.__class__.correct_pretrain_output.shape)

    def test_pretrain_full(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForPretraining(config)
        output = mdl(self.__class__.data)
        self.assertEqual(
            output.prediction_outputs.shape,
            self.__class__.correct_pretrain_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

    def test_pretrain_full_with_return_dict(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForPretraining(config)
        output = mdl(self.__class__.data, return_dict=False)
        self.assertEqual(output[1].shape, self.__class__.correct_pretrain_output.shape)
        self.assertEqual(output[2].shape, self.__class__.enc_output.shape)
        self.assertEqual(output[0].item() < np.inf, True)

    def test_forecast_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerForPredictionHead(
            config=config,
        )
        # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
        output = head(self.__class__.enc_output)

        self.assertEqual(output.shape, self.__class__.correct_forecast_output.shape)

    def check_module(
        self,
        task,
        params=None,
        output_hidden_states=True,
    ):
        config = PatchTSMixerConfig(**params)
        if task == "forecast":
            mdl = PatchTSMixerForPrediction(config)
            target_input = self.__class__.correct_forecast_output
            if config.prediction_channel_indices is not None:
                target_output = self.__class__.correct_sel_forecast_output
            else:
                target_output = target_input
            ref_samples = target_output.unsqueeze(1).expand(-1, config.num_parallel_samples, -1, -1)
            ground_truth_arg = "future_values"
            output_predictions_arg = "prediction_outputs"
        elif task == "classification":
            mdl = PatchTSMixerForTimeSeriesClassification(config)
            target_input = self.__class__.correct_classification_classes
            target_output = self.__class__.correct_classification_output
            ground_truth_arg = "target_values"
            output_predictions_arg = "prediction_outputs"
        elif task == "regression":
            mdl = PatchTSMixerForRegression(config)
            target_input = self.__class__.correct_regression_output
            target_output = self.__class__.correct_regression_output
            ref_samples = target_output.unsqueeze(1).expand(-1, config.num_parallel_samples, -1)
            ground_truth_arg = "target_values"
            output_predictions_arg = "regression_outputs"
        elif task == "pretrain":
            mdl = PatchTSMixerForPretraining(config)
            target_input = None
            target_output = self.__class__.correct_pretrain_output
            ground_truth_arg = None
            output_predictions_arg = "prediction_outputs"
        else:
            print("invalid task")

        enc_output = self.__class__.enc_output

        if target_input is None:
            output = mdl(self.__class__.data, output_hidden_states=output_hidden_states)
        else:
            output = mdl(
                self.__class__.data,
                **{
                    ground_truth_arg: target_input,
                    "output_hidden_states": output_hidden_states,
                },
            )

        prediction_outputs = getattr(output, output_predictions_arg)
        if isinstance(prediction_outputs, tuple):
            for t in prediction_outputs:
                self.assertEqual(t.shape, target_output.shape)
        else:
            self.assertEqual(prediction_outputs.shape, target_output.shape)

        self.assertEqual(output.last_hidden_state.shape, enc_output.shape)

        if output_hidden_states is True:
            self.assertEqual(len(output.hidden_states), params["num_layers"])

        else:
            self.assertEqual(output.hidden_states, None)

        self.assertEqual(output.loss.item() < np.inf, True)

        if config.loss == "nll" and task in ["forecast", "regression"]:
            samples = mdl.generate(self.__class__.data)
            self.assertEqual(samples.sequences.shape, ref_samples.shape)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False],
                [True, False, "mean", "std"],
                [True, False],
                [None, [0, 2]],
                ["mse", "nll"],
            )
        )
    )
    def test_forecast(self, mode, self_attn, scaling, gated_attn, prediction_channel_indices, loss):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            self_attn=self_attn,
            scaling=scaling,
            prediction_channel_indices=prediction_channel_indices,
            gated_attn=gated_attn,
            loss=loss,
        )

        self.check_module(task="forecast", params=params)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False],
                [True, False, "mean", "std"],
                [True, False],
                ["max_pool", "avg_pool"],
            )
        )
    )
    def test_classification(self, mode, self_attn, scaling, gated_attn, head_aggregation):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            self_attn=self_attn,
            scaling=scaling,
            head_aggregation=head_aggregation,
            gated_attn=gated_attn,
        )

        self.check_module(task="classification", params=params)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False],
                [True, False, "mean", "std"],
                [True, False],
                ["max_pool", "avg_pool"],
                ["mse", "nll"],
            )
        )
    )
    def test_regression(self, mode, self_attn, scaling, gated_attn, head_aggregation, loss):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            self_attn=self_attn,
            scaling=scaling,
            head_aggregation=head_aggregation,
            gated_attn=gated_attn,
            loss=loss,
        )

        self.check_module(task="regression", params=params)

    @parameterized.expand(
        list(
            itertools.product(
                ["common_channel", "mix_channel"],
                [True, False],
                [True, False, "mean", "std"],
                [True, False],
                ["random", "forecast"],
                [True, False],
                [True, False],
            )
        )
    )
    def test_pretrain(
        self,
        mode,
        self_attn,
        scaling,
        gated_attn,
        mask_type,
        masked_loss,
        channel_consistent_masking,
    ):
        params = self.__class__.params.copy()
        params.update(
            mode=mode,
            self_attn=self_attn,
            scaling=scaling,
            gated_attn=gated_attn,
            mask_type=mask_type,
            masked_loss=masked_loss,
            channel_consistent_masking=channel_consistent_masking,
        )

        self.check_module(task="pretrain", params=params)

    def forecast_full_module(self, params=None, output_hidden_states=False, return_dict=None):
        config = PatchTSMixerConfig(**params)
        mdl = PatchTSMixerForPrediction(config)

        target_val = self.__class__.correct_forecast_output

        if config.prediction_channel_indices is not None:
            target_val = self.__class__.correct_sel_forecast_output

        enc_output = self.__class__.enc_output

        output = mdl(
            self.__class__.data,
            future_values=self.__class__.correct_forecast_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(output, tuple):
            output = PatchTSMixerForPredictionOutput(*output)

        if config.loss == "mse":
            self.assertEqual(output.prediction_outputs.shape, target_val.shape)

        self.assertEqual(output.last_hidden_state.shape, enc_output.shape)

        if output_hidden_states is True:
            self.assertEqual(len(output.hidden_states), params["num_layers"])

        else:
            self.assertEqual(output.hidden_states, None)

        self.assertEqual(output.loss.item() < np.inf, True)

        if config.loss == "nll":
            samples = mdl.generate(self.__class__.data)
            ref_samples = target_val.unsqueeze(1).expand(-1, params["num_parallel_samples"], -1, -1)
            self.assertEqual(samples.sequences.shape, ref_samples.shape)

    def test_forecast_full(self):
        self.check_module(task="forecast", params=self.__class__.params, output_hidden_states=True)
        # self.forecast_full_module(self.__class__.params, output_hidden_states = True)

    def test_forecast_full_2(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
        )
        self.forecast_full_module(params, output_hidden_states=True)

    def test_forecast_full_2_with_return_dict(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
        )
        self.forecast_full_module(params, output_hidden_states=True, return_dict=False)

    def test_forecast_full_3(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
        )
        self.forecast_full_module(params, output_hidden_states=True)

    def test_forecast_full_5(self):
        params = self.__class__.params.copy()
        params.update(
            self_attn=True,
            use_positional_encoding=True,
            positional_encoding="sincos",
        )
        self.forecast_full_module(params, output_hidden_states=True)

    def test_forecast_full_4(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            prediction_channel_indices=[0, 2],
        )
        self.forecast_full_module(params)

    def test_forecast_full_distributional(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            prediction_channel_indices=[0, 2],
            loss="nll",
            distribution_output="normal",
        )

        self.forecast_full_module(params)

    def test_forecast_full_distributional_2(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            prediction_channel_indices=[0, 2],
            loss="nll",
            # distribution_output = "normal",
        )
        self.forecast_full_module(params)

    def test_forecast_full_distributional_3(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            # prediction_channel_indices=[0, 2],
            loss="nll",
            distribution_output="normal",
        )
        self.forecast_full_module(params)

    def test_forecast_full_distributional_4(self):
        params = self.__class__.params.copy()
        params.update(
            mode="mix_channel",
            # prediction_channel_indices=[0, 2],
            loss="nll",
            distribution_output="normal",
        )
        self.forecast_full_module(params)

    def test_classification_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerLinearHead(
            config=config,
        )
        # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
        output = head(self.__class__.enc_output)

        self.assertEqual(output.shape, self.__class__.correct_classification_output.shape)

    def test_classification_full(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForTimeSeriesClassification(config)
        output = mdl(
            self.__class__.data,
            target_values=self.__class__.correct_classification_classes,
        )
        self.assertEqual(
            output.prediction_outputs.shape,
            self.__class__.correct_classification_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

    def test_classification_full_with_return_dict(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForTimeSeriesClassification(config)
        output = mdl(
            self.__class__.data,
            target_values=self.__class__.correct_classification_classes,
            return_dict=False,
        )
        if isinstance(output, tuple):
            output = PatchTSMixerForTimeSeriesClassificationOutput(*output)
        self.assertEqual(
            output.prediction_outputs.shape,
            self.__class__.correct_classification_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

    def test_regression_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerLinearHead(
            config=config,
        )
        output = head(self.__class__.enc_output)
        self.assertEqual(output.shape, self.__class__.correct_regression_output.shape)

    def test_regression_full(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForRegression(config)
        output = mdl(self.__class__.data, target_values=self.__class__.correct_regression_output)
        self.assertEqual(
            output.regression_outputs.shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

    def test_regression_full_with_return_dict(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForRegression(config)
        output = mdl(
            self.__class__.data,
            target_values=self.__class__.correct_regression_output,
            return_dict=False,
        )
        if isinstance(output, tuple):
            output = PatchTSMixerForRegressionOutput(*output)
        self.assertEqual(
            output.regression_outputs.shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

    def test_regression_full_distribute(self):
        params = self.__class__.params.copy()
        params.update(loss="nll", distribution_output="normal")

        config = PatchTSMixerConfig(**params)

        mdl = PatchTSMixerForRegression(config)
        output = mdl(self.__class__.data, target_values=self.__class__.correct_regression_output)
        self.assertEqual(
            output.regression_outputs[0].shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(
            output.regression_outputs[1].shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

        if config.loss == "nll":
            samples = mdl.generate(self.__class__.data)
            ref_samples = self.__class__.correct_regression_output.unsqueeze(1).expand(
                -1, params["num_parallel_samples"], -1
            )
            self.assertEqual(samples.sequences.shape, ref_samples.shape)

    def test_regression_full_distribute_2(self):
        params = self.__class__.params.copy()
        params.update(loss="nll", distribution_output="student_t")

        config = PatchTSMixerConfig(**params)

        mdl = PatchTSMixerForRegression(config)
        output = mdl(self.__class__.data, target_values=self.__class__.correct_regression_output)
        self.assertEqual(
            output.regression_outputs[0].shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(
            output.regression_outputs[1].shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)
        self.assertEqual(output.loss.item() < np.inf, True)

        if config.loss == "nll":
            samples = mdl.generate(self.__class__.data)
            ref_samples = self.__class__.correct_regression_output.unsqueeze(1).expand(
                -1, params["num_parallel_samples"], -1
            )
            self.assertEqual(samples.sequences.shape, ref_samples.shape)
