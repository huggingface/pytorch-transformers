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
"""Testing suite for the PyTorch VITS model."""

import copy
import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from transformers import PretrainedConfig, VitsConfig
from transformers.testing_utils import (
    is_flaky,
    is_torch_available,
    require_torch,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    global_rng,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import VitsModel, VitsTokenizer


CONFIG_NAME = "config.json"
GENERATION_CONFIG_NAME = "generation_config.json"


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


@require_torch
class VitsModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=False,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        flow_size=16,
        vocab_size=38,
        spectrogram_bins=8,
        duration_predictor_num_flows=2,
        duration_predictor_filter_channels=16,
        prior_encoder_num_flows=2,
        upsample_initial_channel=16,
        upsample_rates=[8, 2],
        upsample_kernel_sizes=[16, 4],
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.flow_size = flow_size
        self.vocab_size = vocab_size
        self.spectrogram_bins = spectrogram_bins
        self.duration_predictor_num_flows = duration_predictor_num_flows
        self.duration_predictor_filter_channels = duration_predictor_filter_channels
        self.prior_encoder_num_flows = prior_encoder_num_flows
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(2)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return VitsConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            flow_size=self.flow_size,
            vocab_size=self.vocab_size,
            spectrogram_bins=self.spectrogram_bins,
            duration_predictor_num_flows=self.duration_predictor_num_flows,
            prior_encoder_num_flows=self.prior_encoder_num_flows,
            duration_predictor_filter_channels=self.duration_predictor_filter_channels,
            posterior_encoder_num_wavenet_layers=self.num_hidden_layers,
            upsample_initial_channel=self.upsample_initial_channel,
            upsample_rates=self.upsample_rates,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = VitsModel(config=config).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        result = model(input_ids, attention_mask=attention_mask)
        self.parent.assertEqual((self.batch_size, 624), result.waveform.shape)


@require_torch
class VitsModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (VitsModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": VitsModel, "text-to-audio": VitsModel} if is_torch_available() else {}
    )
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    has_attentions = False

    def setUp(self):
        self.model_tester = VitsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VitsConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    # TODO: @ydshieh
    @is_flaky(description="torch 2.2.0 gives `Timeout >120.0s`")
    def test_pipeline_feature_extraction(self):
        super().test_pipeline_feature_extraction()

    @is_flaky(description="torch 2.2.0 gives `Timeout >120.0s`")
    def test_pipeline_feature_extraction_fp16(self):
        super().test_pipeline_feature_extraction_fp16()

    @unittest.skip(reason="Need to fix this after #26538")
    def test_model_forward(self):
        set_seed(12345)
        global_rng.seed(12345)
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    @require_torch_multi_gpu
    # override to force all elements of the batch to have the same sequence length across GPUs
    def test_multi_gpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.use_stochastic_duration_prediction = False

        # move input tensors to cuda:O
        for key, value in inputs_dict.items():
            if torch.is_tensor(value):
                # make all elements of the batch the same -> ensures the output seq lengths are the same for DP
                value[1:] = value[0]
                inputs_dict[key] = value.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = torch.nn.DataParallel(model)
            set_seed(555)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class)).waveform

    @unittest.skip(reason="VITS is not deterministic")
    def test_determinism(self):
        pass

    @unittest.skip(reason="VITS is not deterministic")
    def test_batching_equivalence(self):
        pass

    @is_flaky(
        max_attempts=3,
        description="Weight initialisation for the VITS conv layers sometimes exceeds the kaiming normal range",
    )
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        uniform_init_parms = [
            "emb_rel_k",
            "emb_rel_v",
            "conv_1",
            "conv_2",
            "conv_pre",
            "conv_post",
            "conv_proj",
            "conv_dds",
            "project",
            "wavenet.in_layers",
            "wavenet.res_skip_layers",
            "upsampler",
            "resblocks",
        ]

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @unittest.skip(reason="VITS has no inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VITS has no input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override since the model is not deterministic, so we need to set the seed for each forward pass
    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                set_seed(0)
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                set_seed(0)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

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
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
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
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    # override since the model is not deterministic, so we need to set the seed for each forward pass
    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                set_seed(0)
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    set_seed(0)
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)


@require_torch
@slow
class VitsModelIntegrationTests(unittest.TestCase):
    def test_forward(self):
        # GPU gives different results than CPU
        torch_device = "cpu"

        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        model.to(torch_device)

        tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")

        set_seed(555)  # make deterministic

        input_text = "Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel!"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.waveform.shape, (1, 87040))
        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                -0.0042,  0.0176,  0.0354,  0.0504,  0.0621,  0.0777,  0.0980,  0.1224,
                 0.1475,  0.1679,  0.1817,  0.1832,  0.1713,  0.1542,  0.1384,  0.1256,
                 0.1147,  0.1066,  0.1026,  0.0958,  0.0823,  0.0610,  0.0340,  0.0022,
                -0.0337, -0.0677, -0.0969, -0.1178, -0.1311, -0.1363
            ]
        )
        # fmt: on
        torch.testing.assert_close(outputs.waveform[0, 10000:10030].cpu(), EXPECTED_LOGITS, rtol=1e-4, atol=1e-4)
