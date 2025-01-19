# coding=utf-8
# Copyright 2024 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""TF 2.0  Mistral model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    get_tf_activation,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_mistral import MistralConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


def _make_causal_mask(input_ids_shape, dtype, past_key_values_length=0):
    """
    Make causal mask used for bi-directional self-attention, supporting both static and dynamic shapes.
    """
    bsz, tgt_len = input_ids_shape

    # Create a matrix where only the lower triangle and diagonal are filled with zeros (causal mask)
    mask = tf.fill((tgt_len, tgt_len), tf.dtypes.as_dtype(dtype).min)
    mask_cond = tf.range(tgt_len)
    mask = tf.where(mask_cond[:, None] >= mask_cond[None, :], 0.0, mask)

    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    if bsz is None:
        # When batch size is dynamic, expand and tile
        # so we can compile a functional model
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, 0)  # shape: (1, 1, tgt_len, tgt_len + past_key_values_length)
        mask = tf.tile(mask, [bsz, 1, 1, 1])
    else:
        # When batch size is static, directly use broadcast_to
        mask = tf.broadcast_to(mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length))

    return mask


def _expand_mask(mask, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = shape_list(mask)
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
    expanded_mask = tf.broadcast_to(expanded_mask, [bsz, 1, tgt_len, src_len])

    inverted_mask = 1.0 - tf.cast(expanded_mask, dtype)

    return tf.where(
        tf.cast(inverted_mask, bool), tf.fill(dims=shape_list(inverted_mask), value=tf.float32.min), inverted_mask
    )


class TFMistralRMSNorm(keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        """
        TFMistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def build(self, input_shape=None):
        self.weight = self.add_weight(
            name="weight",
            shape=self.hidden_size,
            initializer="ones",
        )
        if self.built:
            return
        self.built = True

    def call(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = tf.cast(hidden_states, tf.float32)
        variance = tf.reduce_mean(tf.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = tf.divide(hidden_states, tf.sqrt(variance + self.variance_epsilon))
        return self.weight * tf.cast(hidden_states, input_dtype)


# Verification: https://colab.research.google.com/gist/ariG23498/f8d8131b795a131b93d99e70ee93c192/scratchpad.ipynb
class TFMistralRotaryEmbedding(keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (tf.range(start=0, limit=self.dim, delta=2, dtype=tf.float32) / self.dim))

    def call(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        t = tf.cast(tf.range(seq_len, dtype=tf.int64), self.inv_freq.dtype)
        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        emb = tf.concat([freqs, freqs], axis=-1)
        cos_values = tf.cast(tf.cos(emb), x.dtype)
        sin_values = tf.cast(tf.sin(emb), x.dtype)

        cos_values = cos_values[:seq_len]
        cos_values = tf.cast(cos_values, dtype=x.dtype)
        sin_values = sin_values[:seq_len]
        sin_values = tf.cast(sin_values, dtype=x.dtype)
        return (cos_values, sin_values)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    mid_length = shape_list(x)[-1] // 2
    x1 = x[..., :mid_length]
    x2 = x[..., mid_length:]
    return tf.concat([-x2, x1], axis=-1)


# Verification: https://colab.research.google.com/gist/ariG23498/bb8474baeb33f4ae6ed7d77da5f7e7a4/scratchpad.ipynb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`tf.Tensor`): The query tensor.
        k (`tf.Tensor`): The key tensor.
        cos (`tf.Tensor`): The cosine part of the rotary embedding.
        sin (`tf.Tensor`): The sine part of the rotary embedding.
        position_ids (`tf.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(tf.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = tf.expand_dims(tf.gather(cos, position_ids), unsqueeze_dim)
    sin = tf.expand_dims(tf.gather(sin, position_ids), unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TFMistralMLP(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = keras.layers.Dense(self.intermediate_size, use_bias=False, name="gate_proj")
        self.up_proj = keras.layers.Dense(self.intermediate_size, use_bias=False, name="up_proj")
        self.down_proj = keras.layers.Dense(self.hidden_size, use_bias=False, name="down_proj")
        self.act_fn = get_tf_activation(config.hidden_act)

    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "gate_proj", None) is not None:
            with tf.name_scope(self.gate_proj.name):
                self.gate_proj.build((self.hidden_size,))
        if getattr(self, "up_proj", None) is not None:
            with tf.name_scope(self.up_proj.name):
                self.up_proj.build((self.hidden_size,))
        if getattr(self, "down_proj", None) is not None:
            with tf.name_scope(self.down_proj.name):
                self.down_proj.build((self.intermediate_size,))


# Verification: https://colab.research.google.com/gist/ariG23498/556d443d491966763ce2e7eee336efed/scratchpad.ipynb
def repeat_kv(hidden_states: tf.Tensor, n_rep: int) -> tf.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = shape_list(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = tf.expand_dims(hidden_states, 2)
    hidden_states = tf.repeat(hidden_states, repeats=n_rep, axis=2)
    return tf.reshape(hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim))


class TFMistralAttention(keras.layers.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = keras.layers.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = keras.layers.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name="v_proj")
        self.o_proj = keras.layers.Dense(self.hidden_size, use_bias=False, name="o_proj")

        self.rotary_emb = TFMistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            name="rotary_emb",
        )
        self.dropout = keras.layers.Dropout(rate=self.attention_dropout)

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        tensor = tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        tensor = tf.transpose(tensor, perm=(0, 2, 1, 3))
        return tensor

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training=None,
        **kwargs,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = shape_list(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = tf.transpose(
            tf.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim)), perm=(0, 2, 1, 3)
        )
        key_states = tf.transpose(
            tf.reshape(key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)), perm=(0, 2, 1, 3)
        )
        value_states = tf.transpose(
            tf.reshape(value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)), perm=(0, 2, 1, 3)
        )

        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(
            x=value_states,
            seq_len=kv_seq_len,
        )
        query_states, key_states = apply_rotary_pos_emb(
            q=query_states,
            k=key_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )

        if past_key_value is not None:
            # resue k, v, self_attention
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = tf.matmul(query_states, key_states, transpose_b=True) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, query_states.dtype)
        attn_weights = self.dropout(
            attn_weights,
            training=training,
        )
        attn_output = tf.matmul(attn_weights, value_states)

        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build((self.hidden_size,))
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build((self.hidden_size,))
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build((self.hidden_size,))
        if getattr(self, "o_proj", None) is not None:
            with tf.name_scope(self.o_proj.name):
                self.o_proj.build((self.num_heads * self.head_dim,))


class TFMistralDecoderLayer(keras.layers.Layer):
    def __init__(self, config: MistralConfig, layer_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size

        self.self_attn = TFMistralAttention(config, layer_idx, name="self_attn")

        self.mlp = TFMistralMLP(config, name="mlp")
        self.input_layernorm = TFMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFMistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, name="post_attention_layernorm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[tf.Tensor, Optional[Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(tf.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, "input_layernorm", None) is not None:
            with tf.name_scope(self.input_layernorm.name):
                self.input_layernorm.build(None)
        if getattr(self, "post_attention_layernorm", None) is not None:
            with tf.name_scope(self.post_attention_layernorm.name):
                self.post_attention_layernorm.build(None)


@keras_serializable
class TFMistralMainLayer(keras.layers.Layer):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    config_class = MistralConfig

    def __init__(self, config: MistralConfig, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # TF and PT Embedding check: https://colab.research.google.com/gist/ariG23498/2b9826818875c9c4968c79cb19f55f2c/scratchpad.ipynb
        self.embed_tokens = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            name="embed_tokens",
        )
        self.layers = [
            TFMistralDecoderLayer(config, layer_idx, name=f"layers.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ]
        self._attn_implementation = config._attn_implementation
        self.norm = TFMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="norm")
        self.config = config

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        # if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = shape_list(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = shape_list(inputs_embeds)
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = shape_list(past_key_values[0][0])[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = tf.range(
                start=past_key_values_length, limit=seq_length + past_key_values_length, dtype=tf.int64
            )
            position_ids = tf.reshape(tf.expand_dims(position_ids, 0), (-1, seq_length))

        else:
            position_ids = tf.cast(tf.reshape(position_ids, (-1, seq_length)), tf.int64)

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = tf.ones((batch_size, seq_length_with_past), dtype=tf.bool)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build(None)
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


MISTRAL_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `model` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`MistralConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class TFMistralPreTrainedModel(TFPreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"


MISTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(tf.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            One formats is allowed:
            - Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class TFMistralModel(TFMistralPreTrainedModel):
    def __init__(self, config: MistralConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFMistralMainLayer(config, name="model")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)


class TFMistralForCausalLM(TFMistralPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFMistralMainLayer(config, name="model")
        self.vocab_size = config.vocab_size
        self.lm_head = keras.layers.Dense(
            config.vocab_size,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="lm_head",
        )
        self.config = config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
                or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = tf.cast(logits, tf.float32)

        loss = None
        if labels is not None:
            # shift labels to the left and cut last logit token
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values:
            input_ids = tf.expand_dims(input_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build((self.config.hidden_size,))


@add_start_docstrings(
    """
    The Mistral Model transformer with a sequence classification head on top (linear layer).

    [`MistralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MISTRAL_START_DOCSTRING,
)
class TFMistralForSequenceClassification(TFMistralPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.model = TFMistralMainLayer(config, name="model")
        self.score = keras.layers.Dense(
            self.num_labels,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
        )
        self.config = config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFSequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        logits_shape = shape_list(logits)
        in_logits = None

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                sequence_lengths = tf.where(
                    sequence_lengths >= 0,
                    sequence_lengths,
                    tf.cast(shape_list(input_ids[-1]), sequence_lengths.dtype) - 1,
                )
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                sequence_lengths = -1
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        if labels is not None:
            if self.config.pad_token_id is None and logits_shape[0] != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0 : logits_shape[0], sequence_lengths]

            loss = self.hf_compute_loss(tf.reshape(labels, [-1]), tf.reshape(in_logits, [-1, self.num_labels]))
        pooled_logits = in_logits if in_logits is not None else logits

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, "score", None) is not None:
            with tf.name_scope(self.score.name):
                self.score.build((self.config.hidden_size,))


__all__ = ['TFMistralModel', 'TFMistralForCausalLM', 'TFMistralForSequenceClassification', 'TFMistralPreTrainedModel']