# coding=utf-8
# Copyright 2021 The Marian Team Authors and The Google Flax Team Authors And The HuggingFace Inc. team. All rights reserved.
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
"""Flax Marian model."""

import math
import random
from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_marian import MarianConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Helsinki-NLP/opus-mt-en-de"
_CONFIG_FOR_DOC = "MarianConfig"


MARIAN_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`MarianConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

MARIAN_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


MARIAN_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MARIAN_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def create_sinusoidal_positions(n_pos, dim):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    sentinel = dim // 2 + dim % 2
    out = np.zeros_like(position_enc)
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])

    return jnp.array(out)


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = jnp.zeros_like(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention with Bart->Marian
class FlaxMarianAttention(nn.Module):
    config: MarianConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()

        self.dropout_layer = nn.Dropout(rate=self.dropout)

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # handle cache prepare causal attention mask
        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # combine masks if needed
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayer with Bart->Marian
class FlaxMarianEncoderLayer(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        self.fc1 = nn.Dense(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartEncoderLayerCollection with Bart->Marian
class FlaxMarianEncoderLayerCollection(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxMarianEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]
        self.layerdrop = self.config.encoder_layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    deterministic,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayer with Bart->Marian
class FlaxMarianDecoderLayer(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
        )
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.encoder_attn = FlaxMarianAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std)
        )
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderLayerCollection with Bart->Marian
class FlaxMarianDecoderLayerCollection(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxMarianDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]
        self.layerdrop = self.config.decoder_layerdrop

    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None)
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    deterministic=deterministic,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxMarianEncoder(nn.Module):
    config: MarianConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        embed_dim = self.config.d_model
        self.max_source_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        self.layers = FlaxMarianEncoderLayerCollection(self.config, self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # explicitly cast the positions here, since self.embed_positions are not registered as parameters
        positions = positions.astype(inputs_embeds.dtype)

        hidden_states = inputs_embeds + positions
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs

        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxMarianDecoder(nn.Module):
    config: MarianConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        embed_dim = self.config.d_model
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0

        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, embed_dim)
        self.layers = FlaxMarianDecoderLayerCollection(self.config, self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed positions
        positions = jnp.take(self.embed_positions, position_ids, axis=0)
        # explicitly cast the positions here, since self.embed_positions are not registered as parameters
        positions = positions.astype(inputs_embeds.dtype)

        hidden_states = inputs_embeds + positions

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxMarianModule(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.encoder = FlaxMarianEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.decoder = FlaxMarianDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxMarianPreTrainedModel(FlaxPreTrainedModel):
    config_class = MarianConfig
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: MarianConfig,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # make sure initialization pass will work for FlaxMarianForSequenceClassificationModule
        input_ids = input_ids.at[(..., -1)].set(self.config.eos_token_id)
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = input_ids
        decoder_attention_mask = jnp.ones_like(input_ids)

        batch_size, sequence_length = input_ids.shape
        position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        # init input variables to retrieve cache
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape
        )

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(MARIAN_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=MarianConfig)
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxMarianMTModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=64, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, position_ids, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, position_ids, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=MarianConfig)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxMarianMTModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=64, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxMarianAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # prepare decoder inputs
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


@add_start_docstrings(
    "The bare Marian Model transformer outputting raw hidden-states without any specific head on top.",
    MARIAN_START_DOCSTRING,
)
class FlaxMarianModel(FlaxMarianPreTrainedModel):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    module_class = FlaxMarianModule


append_call_sample_docstring(FlaxMarianModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


class FlaxMarianMTModule(nn.Module):
    config: MarianConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.model = FlaxMarianModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings))

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits += self.final_logits_bias.astype(self.dtype)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    "The MARIAN Model with a language modeling head. Can be used for translation.", MARIAN_START_DOCSTRING
)
class FlaxMarianMTModel(FlaxMarianPreTrainedModel):
    module_class = FlaxMarianMTModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(MARIAN_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MarianConfig)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxMarianMTModel

        >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=64, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxMarianAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if self.config.tie_word_embeddings:
                shared_embedding = module.model.variables["params"]["shared"]["embedding"]
                lm_logits = module.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            else:
                lm_logits = module.lm_head(hidden_states)
            lm_logits += module.final_logits_bias.astype(self.dtype)

            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def _adapt_logits_for_beam_search(self, logits):
        """This function enforces the padding token never to be generated."""
        logits = logits.at[:, :, self.config.pad_token_id].set(float("-inf"))
        return logits

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs


FLAX_MARIAN_MT_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxMarianMTModel

    >>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    >>> text = "My friends are cool but they eat too many carbs."
    >>> input_ids = tokenizer(text, max_length=64, return_tensors="jax").input_ids

    >>> sequences = model.generate(input_ids, max_length=64, num_beams=2).sequences

    >>> outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    >>> # should give *Meine Freunde sind cool, aber sie essen zu viele Kohlenhydrate.*
    ```
"""

overwrite_call_docstring(
    FlaxMarianMTModel,
    MARIAN_INPUTS_DOCSTRING + FLAX_MARIAN_MT_DOCSTRING,
)
append_replace_return_docstrings(FlaxMarianMTModel, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)


__all__ = ["FlaxMarianModel", "FlaxMarianMTModel", "FlaxMarianPreTrainedModel"]
