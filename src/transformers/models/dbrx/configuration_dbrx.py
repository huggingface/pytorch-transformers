# coding=utf-8
# Copyright 2024 Databricks Mosaic Research and The HuggingFace Inc. team. All rights reserved.
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
"""DBRX model configuration"""

from typing import Any, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class DbrxAttentionConfig(PretrainedConfig):
    """Configuration class for Dbrx Attention.

    [`DbrxAttention`] class. It is used to instantiate attention layers
    according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        clip_qkv (`float`, *optional*):
            If set, clip the queries, keys, and values in the attention layer to this value.
        kv_n_heads (`int`, *optional*, defaults to 1): For grouped_query_attention only, allow user to specify number of kv heads.
        rope_theta (`float`, *optional*, defaults to 10000.0): The base frequency for rope.
    """

    base_config_key = "attn_config"

    def __init__(
        self,
        attn_pdrop: float = 0.0,
        clip_qkv: Optional[float] = None,
        kv_n_heads: int = 1,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta

        for k in ["model_type", "attn_implementation", "transformers_version", "_commit_hash", "torch_dtype"]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")


class DbrxFFNConfig(PretrainedConfig):
    """Configuration class for Dbrx FFN.

    [`DbrxFFN`] class. It is used to instantiate feedforward layers according to
    the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        ffn_act_fn (`dict`, *optional*, defaults to `None`): A dict specifying activation function for the FFN.
            The dict should have a key 'name' with the value being the name of the activation function along with
            any additional keyword arguments. If `None`, then set to `{"name": "silu"}`.
        ffn_hidden_size (`int`, *optional*, defaults to 3584): The hidden size of the feedforward network.
        moe_num_experts (`int`, *optional*, defaults to 4): The number of experts in the mixture of experts layer.
        moe_top_k (`int`, *optional*, defaults to 1): The number of experts to use in the mixture of experts layer.
        moe_jitter_eps (`float`, *optional*, defaults to `None`): If not `None`, the jitter epsilon for the mixture of experts layer.
        moe_loss_weight (`float`, *optional*, defaults to 0.01): The loss weight for the mixture of experts layer.
        moe_normalize_expert_weights (`float`, *optional*, defaults to 1.0): The normalization factor for the expert weights.
    """

    base_config_key = "ffn_config"

    def __init__(
        self,
        ffn_act_fn: dict = None,
        ffn_hidden_size: int = 3584,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_jitter_eps: Optional[float] = None,
        moe_loss_weight: float = 0.01,
        moe_normalize_expert_weights: Optional[float] = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        if ffn_act_fn is None:
            ffn_act_fn = {"name": "silu"}
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights

        for k in ["model_type", "attn_implementation", "transformers_version", "_commit_hash", "torch_dtype"]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")


class DbrxConfig(PretrainedConfig):
    r"""

    This is the configuration class to store the configuration of a [`DbrxModel`]. It is used to instantiate a Dbrx model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a different configuration to that of the [databricks/dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 2048):
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        max_seq_len (`int`, *optional*, defaults to 2048):
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Dbrx model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DbrxModel`].
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        attn_config (`dict`, *optional*):
            A dictionary used to configure the model's attention module.
        ffn_config (`dict`, *optional*):
            A dictionary used to configure the model's FFN module.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details.


    Example:
    ```python
    >>> from transformers import DbrxConfig, DbrxModel

    >>> # Initializing a Dbrx configuration
    >>> configuration = DbrxConfig(n_layers=2, d_model=256, n_heads=8, vocab_size=128)

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DbrxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dbrx"
    sub_configs = {"attn_config": DbrxAttentionConfig, "ffn_config": DbrxFFNConfig}
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "max_position_embeddings": "max_seq_len",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        max_seq_len: int = 2048,
        vocab_size: int = 32000,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_config: Optional[DbrxAttentionConfig] = None,
        ffn_config: Optional[DbrxFFNConfig] = None,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        output_router_logits: bool = False,
        **kwargs: Any,
    ):
        if attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        if ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**ffn_config)
        else:
            self.ffn_config = ffn_config

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.output_router_logits = output_router_logits
        self.num_key_value_heads = self.attn_config.kv_n_heads

        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        if tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported for DBRX models.")

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["DbrxConfig"]
