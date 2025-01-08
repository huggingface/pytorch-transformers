# coding=utf-8
# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
"""DeBERTa-v2 model configuration"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType


logger = logging.get_logger(__name__)


class DebertaV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaV2Model`]. It is used to instantiate a
    DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 128100):
            Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DebertaV2Model`].
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 24):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
            are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 0):
            The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        relative_attention (`bool`, *optional*, defaults to `True`):
            Whether use relative position encoding.
        max_relative_positions (`int`, *optional*, defaults to -1):
            The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
            as `max_position_embeddings`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (`bool`, *optional*, defaults to `True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (`List[str]`, *optional*):
            The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
            `["p2c", "c2p"]`, `["p2c", "c2p"]`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        legacy (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the legacy `LegacyDebertaOnlyMLMHead`, which does not work properly
            for mask infilling tasks.

    Example:

    ```python
    >>> from transformers import DebertaV2Config, DebertaV2Model

    >>> # Initializing a DeBERTa-v2 microsoft/deberta-v2-xlarge style configuration
    >>> configuration = DebertaV2Config()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-v2-xlarge style configuration
    >>> model = DebertaV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deberta-v2"

    def __init__(
        self,
        vocab_size=128100,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        intermediate_size=6144,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        legacy=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if isinstance(pos_att_type, str):
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
        self.legacy = legacy


class DebertaV2OnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        if self._config.type_vocab_size > 0:
            return OrderedDict(
                [("input_ids", dynamic_axis), ("attention_mask", dynamic_axis), ("token_type_ids", dynamic_axis)]
            )
        else:
            return OrderedDict([("input_ids", dynamic_axis), ("attention_mask", dynamic_axis)])

    @property
    def default_onnx_opset(self) -> int:
        return 12

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        dummy_inputs = super().generate_dummy_inputs(preprocessor=preprocessor, framework=framework)
        if self._config.type_vocab_size == 0 and "token_type_ids" in dummy_inputs:
            del dummy_inputs["token_type_ids"]
        return dummy_inputs


__all__ = ["DebertaV2Config", "DebertaV2OnnxConfig"]
