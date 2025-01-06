# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
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
"""Mpt configuration"""

from typing import TYPE_CHECKING, Optional, Union


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MptAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture. Most of the arguments are kept for backward
    compatibility with previous MPT models that are hosted on the Hub (previously with `trust_remote_code=True`).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_type (`str`, *optional*, defaults to `"multihead_attention"`):
            type of attention to use. Options: `"multihead_attention"`, `"multiquery_attention"`.
        attn_pdrop (`float`, *optional*, defaults to `0.0`):
            The dropout probability for the attention layers.
        attn_impl (`str`, *optional*, defaults to `"torch"`):
            The attention implementation to use. One of `"torch"`, `"flash"`, or `"triton"`.
        clip_qkv (`float`, *optional*):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        softmax_scale (`float`, *optional*):
            If not `None`, scale the softmax in the attention layer by this value. If `None`, will default to
            `1/sqrt(hidden_size)`.
        prefix_lm (`bool`, *optional*, defaults to `False`):
            Whether the model should operate as a Prefix LM. This requires passing an extra `prefix_mask` argument
            which indicates which tokens belong to the prefix. Tokens in the prefix can attend to one another
            bi-directionally. Tokens outside the prefix use causal attention.
        qk_ln (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to the queries and keys in the attention layer.
        attn_uses_sequence_id (`bool`, *optional*, defaults to `False`):
            Whether to restrict attention to tokens that have the same token_type_ids. When the model is in `train`
            mode, this requires passing an extra *token_type_ids* argument which indicates which sub-sequence each
            token belongs to. Defaults to `False` meaning any provided *token_type_ids* will be ignored.
        alibi (`bool`, *optional*, defaults to `True`):
            Whether or not to use the alibi bias instead of positional embedding.
        alibi_bias_max (`int`, *optional*, defaults to 8):
            The maximum value of the alibi bias.
    """

    base_config_key = "attn_config"

    def __init__(
        self,
        attn_type="multihead_attention",
        attn_pdrop=0,
        attn_impl="torch",
        clip_qkv=None,
        softmax_scale=None,
        prefix_lm=False,
        qk_ln=False,
        attn_uses_sequence_id=False,
        alibi=True,
        alibi_bias_max=8,
        **kwargs,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.attn_pdrop = attn_pdrop
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.softmax_scale = softmax_scale
        self.prefix_lm = prefix_lm
        self.attn_uses_sequence_id = attn_uses_sequence_id
        self.alibi = alibi
        self.qk_ln = qk_ln
        self.alibi_bias_max = alibi_bias_max

        if attn_type not in ["multihead_attention", "multiquery_attention"]:
            raise ValueError(
                f"`attn_type` has to be either `multihead_attention` or `multiquery_attention`. Received: {attn_type}"
            )


class MptConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Mpt-7b architecture
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 2048):
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        expansion_ratio (`int`, *optional*, defaults to 4):
            The ratio of the up/down scale in the MLP.
        max_seq_len (`int`, *optional*, defaults to 2048):
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 50368):
            Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MptModel`]. Check [this
            discussion](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        learned_pos_emb (`bool`, *optional*, defaults to `True`):
            Whether to use learned positional embeddings.
        attn_config (`dict`, *optional*):
            A dictionary used to configure the model's attention module.
        init_device (`str`, *optional*, defaults to `"cpu"`):
            The device to use for parameter initialization. Defined for backward compatibility
        logit_scale (`float`, *optional*):
            If not None, scale the logits by this value.
        no_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in all linear layers.
        verbose (`int`, *optional*, defaults to 0):
            The verbosity level to use for logging. Used in the previous versions of MPT models for logging. This
            argument is deprecated.
        embedding_fraction (`float`, *optional*, defaults to 1.0):
            The fraction to scale the gradients of the embedding layer by.
        norm_type (`str`, *optional*, defaults to `"low_precision_layernorm"`):
            Type of layer norm to use. All MPT models uses the same layer norm implementation. Defined for backward
            compatibility.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import MptConfig, MptModel

    >>> # Initializing a Mpt configuration
    >>> configuration = MptConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MptModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mpt"
    sub_configs = {"attn_config": MptAttentionConfig}
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: MptAttentionConfig = None,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = True,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        initializer_range=0.02,
        **kwargs,
    ):
        if attn_config is None:
            self.attn_config = MptAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = MptAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


__all__ = ["MptConfig"]
