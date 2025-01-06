# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Flax whisper model."""

import math
import random
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey

from ...generation.flax_logits_process import FlaxWhisperTimeStampLogitsProcessor
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
    FlaxSequenceClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"
_CONFIG_FOR_DOC = "WhisperConfig"

remat = nn_partitioning.remat


def sinusoidal_embedding_init(key, shape, dtype=jnp.float_) -> jax.Array:
    """Returns sinusoids for positional embedding"""
    length, channels = shape
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length).reshape(-1, 1) * inv_timescales.reshape(1, -1)
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1).astype(dtype)


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`WhisperConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision
            inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.** If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`]
            and [`~FlaxPreTrainedModel.to_bf16`].
"""

WHISPER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids) Whisper uses the `decoder_start_token_id` as
            the starting token for `decoder_input_ids` generation.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not use `position_ids` in the encoder as `input_features` is always the same size and doesn't
            use masking, but this argument is preserved for compatibility. By default the silence in the input log mel
            spectrogram are ignored.
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

WHISPER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`].
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids)
        encoder_outputs (`tuple(tuple(numpy.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
           Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, numpy.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
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


class FlaxWhisperAttention(nn.Module):
    config: WhisperConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

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
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.q_proj = dense(use_bias=self.bias)
        self.k_proj = dense(use_bias=False)
        self.v_proj = dense(use_bias=self.bias)
        self.out_proj = dense(use_bias=self.bias)

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_target_positions), dtype="bool"), dtype="bool"
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
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

    def _split_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
            # causal mask for cached decoder self-attention: our single query position should only
            # attend to those key positions that have already been generated and cached, not the
            # remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask


# Copied from transformers.models.mbart.modeling_flax_mbart.FlaxMBartEncoderLayer with MBart->Whisper
class FlaxWhisperEncoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
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
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlaxWhisperEncoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        if self.gradient_checkpointing:
            FlaxWhisperEncoderCheckpointLayer = remat(FlaxWhisperEncoderLayer, static_argnums=(2, 3))
            self.layers = [
                FlaxWhisperEncoderCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.encoder_layers)
            ]
        else:
            self.layers = [
                FlaxWhisperEncoderLayer(self.config, name=str(i), dtype=self.dtype)
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


# Copied from transformers.models.mbart.modeling_flax_mbart.FlaxMBartDecoderLayer with MBart->Whisper
class FlaxWhisperDecoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
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
        self.encoder_attn = FlaxWhisperAttention(
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
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache
        )
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class FlaxWhisperDecoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        if self.gradient_checkpointing:
            FlaxWhisperDecoderCheckpointLayer = remat(FlaxWhisperDecoderLayer, static_argnums=(4, 5, 6))
            self.layers = [
                FlaxWhisperDecoderCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.decoder_layers)
            ]
        else:
            self.layers = [
                FlaxWhisperDecoderLayer(self.config, name=str(i), dtype=self.dtype)
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
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    init_cache,
                    output_attentions,
                    deterministic,
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


class FlaxWhisperEncoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.conv1 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )
        self.conv2 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            strides=2,
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        self.layers = FlaxWhisperEncoderLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        self.embed_positions = nn.Embed(
            self.config.max_source_positions,
            self.config.d_model,
            dtype=self.dtype,
            embedding_init=sinusoidal_embedding_init,
        )

        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        input_features: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        if input_features.shape[1:] != (self.config.num_mel_bins, self.config.max_source_positions * 2):
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
            )

        input_features = input_features.transpose(0, 2, 1)
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        # freeze the sinusoidal embeddings by stopping the back-prop
        embed_positions = jax.lax.stop_gradient(embed_positions)
        hidden_states = hidden_states + embed_positions

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask=None,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


class FlaxWhisperDecoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.d_model, dtype=self.dtype)
        self.embed_positions = nn.Embed(self.config.max_target_positions, self.config.d_model, dtype=self.dtype)

        self.layers = FlaxWhisperDecoderLayerCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-5)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxWhisperModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.decoder = FlaxWhisperDecoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        encoder_outputs = self.encoder(
            input_features,
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

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder


class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix: str = "model"
    main_input_name = "input_features"
    module_class: nn.Module = None

    def __init__(
        self,
        config: WhisperConfig,
        input_shape: Tuple[int] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        if input_shape is None:
            input_shape = (1, config.num_mel_bins, 2 * config.max_source_positions)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
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

    # Copied from transformers.models.bart.modeling_flax_bart.FlaxBartPreTrainedModel.init_cache with Bart->Whisper
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
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

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

    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=WhisperConfig)
    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_features, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_features, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=WhisperConfig)
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
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import jax.numpy as jnp

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_features = processor(ds[0]["audio"]["array"], return_tensors="np").input_features

        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((input_features.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )

        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxWhisperAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
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

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
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

        # prepare decoder inputs
        if decoder_position_ids is None:
            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                batch_size, sequence_length = decoder_input_ids.shape
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
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
    "The bare Whisper Model transformer outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    module_class = FlaxWhisperModule


append_call_sample_docstring(FlaxWhisperModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


class FlaxWhisperForConditionalGenerationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.model = FlaxWhisperModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_attention_mask: jnp.ndarray = None,
        decoder_position_ids: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.decoder.embed_tokens.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

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


@add_start_docstrings("The Whisper Model with a language modeling head.", WHISPER_START_DOCSTRING)
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig)
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
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
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

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxWhisperAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if self.config.tie_word_embeddings:
                shared_embedding = module.model.decoder.embed_tokens.variables["params"]["embedding"]
                lm_logits = module.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            else:
                lm_logits = module.lm_head(hidden_states)

            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
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

    def generate(
        self,
        input_features,
        generation_config=None,
        logits_processor=None,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            generation_config.return_timestamps = return_timestamps

        if task is not None:
            generation_config.task = task

        if is_multilingual is not None:
            generation_config.is_multilingual = is_multilingual

        if language is not None:
            generation_config.language = language

        if kwargs is not None and "decoder_input_ids" in kwargs:
            decoder_input_length = len(kwargs["decoder_input_ids"])
        else:
            decoder_input_length = 1

        forced_decoder_ids = []

        if hasattr(generation_config, "is_multilingual") and generation_config.is_multilingual:
            if hasattr(generation_config, "language"):
                forced_decoder_ids.append((1, generation_config.lang_to_id[generation_config.language]))
            else:
                forced_decoder_ids.append((1, None))

            if hasattr(generation_config, "task"):
                forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

        if (
            hasattr(generation_config, "return_timestamps") and generation_config.return_timestamps
        ) or return_timestamps:
            logits_processor = [
                FlaxWhisperTimeStampLogitsProcessor(generation_config, self.config, decoder_input_length)
            ]
        else:
            if forced_decoder_ids and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if len(forced_decoder_ids) > 0:
            generation_config.forced_decoder_ids = forced_decoder_ids

        return super().generate(
            input_features,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

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
            position_ids = decoder_attention_mask.cumsum(-1) - 1
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


FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
    >>> input_features = inputs.input_features
    >>> generated_ids = model.generate(input_ids=input_features)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> transcription
    ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ```
"""

overwrite_call_docstring(
    FlaxWhisperForConditionalGeneration, WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING
)
append_replace_return_docstrings(
    FlaxWhisperForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxWhisperForAudioClassificationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.config.is_encoder_decoder = False
        num_layers = self.config.num_hidden_layers + 1
        if self.config.use_weighted_layer_sum:
            self.layer_weights = jnp.repeat(1 / num_layers, num_layers)
        self.projector = nn.Dense(self.config.classifier_proj_size, dtype=self.dtype)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = jnp.stack(encoder_outputs, axis=1)
            norm_weights = jax.nn.softmax(self.layer_weights, axis=-1)
            hidden_states = jnp.sum(hidden_states * jnp.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = jnp.mean(hidden_states, axis=1)

        logits = self.classifier(pooled_output)

        if not return_dict:
            return (logits,) + encoder_outputs[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("The Whisper Model with an audio classification head on top.", WHISPER_START_DOCSTRING)
class FlaxWhisperForAudioClassification(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForAudioClassificationModule
    dtype: jnp.dtype = jnp.float32

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_features=input_features,
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

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
        )


FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> import jax.numpy as jnp
    >>> from transformers import AutoFeatureExtractor, FlaxWhisperForAudioClassification
    >>> from datasets import load_dataset

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    >>> model = FlaxWhisperForAudioClassification.from_pretrained(
    ...     "sanchit-gandhi/whisper-medium-fleurs-lang-id", from_pt=True
    ... )
    >>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True, trust_remote_code=True)

    >>> sample = next(iter(ds))

    >>> inputs = feature_extractor(
    ...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="np"
    ... )
    >>> input_features = inputs.input_features

    >>> logits = model(input_features).logits

    >>> predicted_class_ids = jnp.argmax(logits).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    >>> predicted_label
    'af_za'
    ```
"""

overwrite_call_docstring(
    FlaxWhisperForAudioClassification, WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING
)
append_replace_return_docstrings(
    FlaxWhisperForAudioClassification, output_type=FlaxSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
)


__all__ = [
    "FlaxWhisperForConditionalGeneration",
    "FlaxWhisperModel",
    "FlaxWhisperPreTrainedModel",
    "FlaxWhisperForAudioClassification",
]
