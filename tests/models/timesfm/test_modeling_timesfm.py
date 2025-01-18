# coding=utf-8
# Copyright 2024 Google LLC and HuggingFace Inc. team.
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

import inspect
import unittest
from typing import List

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from transformers import TimesFmConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_fx_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


if is_torch_fx_available():
    pass

if is_torch_available():
    from transformers import TimesFmModelForPrediction

TOLERANCE = 1e-4


class TimesFmModelTester:
    def __init__(
        self,
        parent,
        patch_len: int = 32,
        context_len: int = 512,
        horizon_len: int = 128,
        freq_size: int = 3,
        num_layers: int = 1,
        model_dim: int = 16,
        intermediate_size: int = 32,
        head_dim: int = 8,
        num_heads: int = 2,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        use_positional_embedding: bool = True,
        initializer_factor: float = 0.0,
        is_training: bool = False,
        batch_size: int = 3,
    ):
        self.parent = parent
        self.patch_len = patch_len
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.quantiles = quantiles
        self.pad_val = pad_val
        self.freq_size = freq_size
        self.model_dim = model_dim
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads
        self.tolerance = tolerance
        self.rms_norm_eps = rms_norm_eps
        self.use_positional_embedding = use_positional_embedding
        self.initializer_factor = initializer_factor
        self.is_training = is_training
        self.batch_size = batch_size

        # The size of test input
        self.seq_length = context_len // patch_len
        self.hidden_size = model_dim

    def get_large_model_config(self):
        return TimesFmConfig.from_pretrained("google/timesfm-1.0-200m-pytorch")

    def get_config(self):
        return TimesFmConfig(
            patch_len=self.patch_len,
            context_len=self.context_len,
            horizon_len=self.horizon_len,
            quantiles=self.quantiles,
            pad_val=self.pad_val,
            freq_size=self.freq_size,
            model_dim=self.model_dim,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            tolerance=self.tolerance,
            rms_norm_eps=self.rms_norm_eps,
            use_positional_embedding=self.use_positional_embedding,
            initializer_factor=self.initializer_factor,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        forecast_input = [
            torch.tensor(np.sin(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.cos(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.tan(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
        ]
        frequency_input = torch.tensor([0, 1, 2], dtype=torch.long, device=torch_device)

        return (self.get_config(), torch.stack(forecast_input, dim=0), frequency_input)

    def prepare_config_and_inputs_for_common(self):
        (config, forecast_input, frequency_input) = self.prepare_config_and_inputs()

        inputs_dict = {
            "inputs": forecast_input,
            "freq": frequency_input,
        }
        return config, inputs_dict


@require_torch
class TimesFmModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (TimesFmModelForPrediction,) if is_torch_available() else ()
    all_generative_model_classes = (TimesFmModelForPrediction,) if is_torch_available() else ()
    all_parallelizable_model_classes = ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = False
    is_encoder_decoder = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = TimesFmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimesFmConfig)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TimesFmModelForPrediction(config)
        model.to(torch_device)
        model.eval()
        results = model(**inputs_dict)
        assert results.mean_predictions is not None

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Model does not have head mask")
    def test_headmasking(self):
        pass

    # the main input name is `inputs`
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(TimesFmModelForPrediction, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(TimesFmModelForPrediction.main_input_name, observed_main_input_name)


@require_torch
@slow
class TimesFmModelIntegrationTests(unittest.TestCase):
    @classmethod
    def load_batch(cls, filename="train-batch.pt"):
        file = hf_hub_download(
            repo_id="hf-internal-testing/tourism-monthly-batch", filename=filename, repo_type="dataset"
        )
        batch = torch.load(file, map_location=torch_device)
        return batch

    def test_inference_no_head(self):
        model = TimesFmModelForPrediction.from_pretrained("huggingface/timesfm-tourism-monthly").to(torch_device)
        batch = self.load_batch()
        with torch.no_grad():
            inputs = batch["past_values"]
            output = model(inputs=inputs).last_hidden_state
        self.assertEqual(
            output.shape, torch.Size([64, model.config.context_len // model.config.patch_len, model.config.model_dim])
        )

        expected_slice = torch.tensor(
            [[-4.0141, 3.3141, 1.9321], [-4.9121, 3.1443, 2.0836], [-5.1142, 2.7376, 2.1566]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))
