# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""GPT Neo model configuration"""

from collections import OrderedDict
from typing import Any, Mapping, Optional

from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging


logger = logging.get_logger(__name__)


class GPTNeoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTNeoModel`]. It is used to instantiate a GPT
    Neo model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the GPTNeo
    [EleutherAI/gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT Neo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTNeoModel`]. Vocabulary size of the model. Defines the different
            tokens that can be represented by the *inputs_ids* passed to the forward method of [`GPTNeoModel`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        attention_types (`List`, *optional*, defaults to `[[['global', 'local'], 12]]`):
            The type of attention for each layer in a `List` of the following format `[[["attention_type"],
            num_layerss]]` e.g. for a 24 layer model `[[["global"], 24]]` or `[[["global", "local"], 12]]` Choose the
            value of `attention_type` from `["global", "local"]`
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        window_size (`int`, *optional*, defaults to 256):
            The size of the sliding window for local attention.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        resid_dropout (`float`, *optional*, defaults to 0.0):
            Residual dropout used in the attention pattern.
        embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing token classification, used in the model [`GPTNeoForTokenClassification`]. The
            dropout ratio for the hidden layer.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 50256):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            The id of the end of sentence token in the vocabulary.

    Example:

    ```python
    >>> from transformers import GPTNeoConfig, GPTNeoModel

    >>> # Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration
    >>> configuration = GPTNeoConfig()

    >>> # Initializing a model (with random weights) from the EleutherAI/gpt-neo-1.3B style configuration
    >>> model = GPTNeoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt_neo"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=24,
        attention_types=[[["global", "local"], 12]],
        num_heads=16,
        intermediate_size=None,
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.attention_types = attention_types
        self.attention_layers = self.expand_attention_types_params(attention_types)

        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions


def custom_unfold(input, dimension, size, step):
    """Custom torch.Tensor.unfold implementation to enable the export to ONNX."""
    import torch

    shape = input.size()
    rank = len(shape)
    sizedim = shape[dimension]

    low_indices = torch.arange(0, sizedim, step)
    min_length = torch.div(sizedim - size, step, rounding_mode="floor") + 1
    indices = torch.arange(size) + low_indices[:min_length][:, None]

    s = [slice(None)] * rank
    s[dimension] = indices
    sliced = input[s]

    perm = list(range(0, rank + 1))
    perm.append(perm.pop(dimension + 1))

    return sliced.permute(perm)


def custom_get_block_length_and_num_blocks(seq_length, window_size):
    """
    Custom implementation for GPTNeoAttentionMixin._get_block_length_and_num_blocks to enable the export to ONNX as
    original implementation uses Python variables and control flow.
    """
    import torch

    candidates = torch.arange(1, window_size)
    remainders = torch.remainder(seq_length, candidates)
    divisor_indices = remainders == 0
    divisors = candidates[divisor_indices]
    largest_divisor = torch.max(divisors)
    return largest_divisor, torch.div(seq_length, largest_divisor, rounding_mode="floor")


class GPTNeoOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_attention_heads(self) -> int:
        return self._config.num_heads

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13


__all__ = ["GPTNeoConfig", "GPTNeoOnnxConfig"]
