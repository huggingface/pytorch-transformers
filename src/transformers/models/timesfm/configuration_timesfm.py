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
"""TimesFM model configuration"""

from typing import List, Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging


logger = logging.get_logger(__name__)


class TimesFmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TimesFmModelForPrediction`] or a [`TFTimesFmDecoder`]. It is used to
    instantiate a TimesFM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TimesFM
    [google/timesfm-1.0-200m](https://huggingface.co/google/timesfm-1.0-200m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        patch_len (`int`, *optional*, defaults to 32):
            The length of one patch in the input sequence.
        context_len (`int`, *optional*, defaults to 512):
            The length of the input context.
        horizon_len (`int`, *optional*, defaults to 128):
            The length of the prediction horizon.
        freq_size (`int`, *optional*, defaults to 3):
            The number of frequency embeddings.
        num_layers (`int`, *optional*, defaults to 20):
            Number of Transformer layers.
        model_dim (`int`, *optional*, defaults to 1280):
            Size of the hidden layers in the feed-forward networks.
        intermediate_size (`int`, *optional*, defaults to 1280):
            Dimension of the MLP representations.
        head_dim (`int`, *optional*, defaults to 80):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * head_dim`.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        tolerance (`float`, *optional*, defaults to 1e-06):
            The tolerance for the quantile loss.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
        quantiles (`List[float]`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`):
            The quantiles to predict.
        pad_val (`float`, *optional*, defaults to 1123581321.0):
            The value used to pad the predictions.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention scores.
        use_positional_embedding (`bool`, *optional*, defaults to `True`):
            Whether to add positional embeddings.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    """

    model_type = "timesfm"
    keys_to_ignore_at_inference = []
    attribute_map = {
        "hidden_size": "model_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    is_encoder_decoder = False

    def __init__(
        self,
        patch_len: int = 32,
        context_len: int = 512,
        horizon_len: int = 128,
        freq_size: int = 3,
        num_layers: int = 20,
        model_dim: int = 1280,
        intermediate_size: int = 1280,
        head_dim: int = 80,
        num_heads: int = 16,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        attention_dropout: float = 0.0,
        use_positional_embedding: bool = True,
        initializer_factor: float = 1.0,
        **kwargs,
    ):
        self.patch_len = patch_len
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.quantiles = quantiles
        self.pad_val = pad_val
        self.freq_size = freq_size
        self.model_dim = model_dim
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.tolerance = tolerance
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.use_positional_embedding = use_positional_embedding
        self.initializer_factor = initializer_factor

        super().__init__(
            is_encoder_decoder=self.is_encoder_decoder,
            **kwargs,
        )


class TimesFmOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {
                0: "batch",
                1: "past_decoder_sequence + sequence",
            }
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {
                0: "batch",
                1: "decoder_sequence",
            }

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13
