# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""TF 2.0 Idefics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from ... import TFPreTrainedModel
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import ModelOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    keras_serializable,
    shape_list,
    unpack_inputs,
)
from ...tf_utils import invert_attention_mask, scaled_dot_product_attention
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_idefics import IdeficsConfig
from .perceiver_tf import TFIdeficsPerceiverResampler
from .vision_tf import TFIdeficsVisionTransformer


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "IdeficsConfig"


@dataclass
class TFIdeficsBaseModelOutputWithPast(ModelOutput):
    """
    Base class for Idefics model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(tf.Tensor)`, *optional*):
            Tuple of `tf.Tensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    last_hidden_state: tf.Tensor = None
    past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    image_hidden_states: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFIdeficsCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Idefics causal language model (or autoregressive) outputs.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(tf.Tensor)`, *optional*):
            Tuple of `tf.Tensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    past_key_values: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    image_hidden_states: Optional[Tuple[tf.Tensor]] = None


def expand_inputs_for_generation(
    input_ids,
    expand_size=1,
    is_encoder_decoder=False,
    attention_mask=None,
    encoder_outputs=None,
    **model_kwargs,
):
    expanded_return_idx = tf.reshape(tf.repeat(tf.range(tf.shape(input_ids)[0]), expand_size), [-1])
    input_ids = tf.gather(input_ids, expanded_return_idx)
    model_kwargs["pixel_values"] = model_kwargs.get("pixel_values", None)
    model_kwargs["image_encoder_embeddings"] = model_kwargs.get("image_encoder_embeddings", None)
    model_kwargs["perceiver_embeddings"] = model_kwargs.get("perceiver_embeddings", None)
    model_kwargs["image_attention_mask"] = model_kwargs.get("image_attention_mask", None)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = tf.gather(token_type_ids, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = tf.gather(attention_mask, expanded_return_idx)

    if model_kwargs["image_attention_mask"] is not None:
        model_kwargs["image_attention_mask"] = tf.gather(model_kwargs["image_attention_mask"], expanded_return_idx)

    if model_kwargs["pixel_values"] is not None:
        model_kwargs["pixel_values"] = tf.gather(model_kwargs["pixel_values"], expanded_return_idx)

    elif model_kwargs["image_encoder_embeddings"] is not None:
        model_kwargs["image_encoder_embeddings"] = tf.gather(
            model_kwargs["image_encoder_embeddings"], expanded_return_idx
        )

    elif model_kwargs["perceiver_embeddings"] is not None:
        model_kwargs["perceiver_embeddings"] = tf.gather(model_kwargs["perceiver_embeddings"], expanded_return_idx)

    return input_ids, model_kwargs


def update_model_kwargs_for_generation(outputs, model_kwargs):
    # must have this key set to at least None
    if "past_key_values" in outputs:
        model_kwargs["past_key_values"] = outputs.past_key_values
    else:
        model_kwargs["past_key_values"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = tf.concat([token_type_ids, token_type_ids[:, -1:, ...]], axis=-1)

    # update attention masks
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = tf.concat(
            [attention_mask, tf.ones_like(attention_mask[:, -1:, ...])], axis=-1
        )
    if "image_attention_mask" in model_kwargs:
        image_attention_mask = model_kwargs["image_attention_mask"]
        last_mask = image_attention_mask[:, -1:, ...]
        model_kwargs["image_attention_mask"] = last_mask

    # Get the precomputed image_hidden_states
    model_kwargs["image_hidden_states"] = outputs.image_hidden_states

    return model_kwargs


def prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1:]

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = tf.math.cumsum(tf.cast(attention_mask, dtype=tf.int64), axis=-1) - 1
        position_ids = tf.where(attention_mask == 0, 1, position_ids)
        if past_key_values is not None:
            position_ids = position_ids[:, -1:]

    pixel_values = kwargs.get("pixel_values", None)
    image_encoder_embeddings = kwargs.get("image_encoder_embeddings", None)
    perceiver_embeddings = kwargs.get("perceiver_embeddings", None)
    image_attention_mask = kwargs.get("image_attention_mask", None)
    interpolate_pos_encoding = kwargs.get("interpolate_pos_encoding", False)

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "image_encoder_embeddings": image_encoder_embeddings,
        "perceiver_embeddings": perceiver_embeddings,
        "image_attention_mask": image_attention_mask,
        "interpolate_pos_encoding": interpolate_pos_encoding,
    }


def freeze_model(model, module_exceptions=[]):
    mapping = {
        "LayerNorm": tf.keras.layers.LayerNormalization,
        "Dense": tf.keras.layers.Dense,
        "Embedding": tf.keras.layers.Embedding,
    }
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    if not hasattr(model, "layers"):
        model.trainable = False  # It is just a layer
        return model
    for layer in model.layers:
        if module_exceptions and any(isinstance(layer, t) for t in module_exceptions_mapped):
            layer.trainable = True  # Explicitly setting it to true to avoid any mistakes
        else:
            layer.trainable = False
    return model


class TFIdeficsDecoupledEmbedding(tf.keras.layers.Embedding):
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `tf.keras.layers.Embedding`.
    """

    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze: Optional[bool] = False,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_embeddings (`int`):
                Size of the dictionary of embeddings
            num_additional_embeddings (`int`):
                Number of additional embeddings. Only useful when you `partially_freeze=True`.
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `False`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.

        Note: there are a lot of other parameters to initialize a standard `tf.keras.layers.Embedding` such as `mask_zero`,
        `input_length` or `embeddings_initializer`. We are not supporting these.
        """
        super().__init__(
            input_dim=num_embeddings,
            output_dim=embedding_dim,
            dtype=dtype,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        if partially_freeze:
            self.trainable = False

        if self.num_additional_embeddings > 0:
            self.additional_embedding = tf.keras.layers.Embedding(
                input_dim=self.num_additional_embeddings,
                output_dim=embedding_dim,
                dtype=dtype,
                name="additional_embedding",
            )

    def call(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
           embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return super().call(input_ids)

        # Clone so that we don't modify the original input_ids later on
        input_ids = tf.identity(input_ids)
        additional_vocab_indices = tf.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = tf.gather_nd(input_ids, additional_vocab_indices)
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids = tf.tensor_scatter_nd_update(
            input_ids,
            additional_vocab_indices,
            # tensor filled with 0, having the same length as additional_vocab_indices
            tf.zeros(tf.shape(additional_vocab_indices)[0], dtype=input_ids.dtype),
        )
        full_vector = super().call(input_ids)

        # overwrite the records with high indices
        full_vector = tf.tensor_scatter_nd_update(full_vector, additional_vocab_indices, additional_embeddings)

        return full_vector

    def extra_repr(self) -> str:
        return "num_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.num_embeddings,
            self.num_additional_embeddings,
            self.output_dim,
            self.partially_freeze,
        )


class TFIdeficsDecoupledLinear(tf.keras.layers.Layer):
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `out_additional_features` > 0,
    then it will create `out_additional_features * in_features` additional parameters that are always trained. If
    `out_additional_features=0`, then the module defaults back to the regular behavior of `tf.keras.layers.Dense`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = 0,
        bias: bool = True,
        partially_freeze: bool = True,
        **kwargs,
    ) -> None:
        """
        out_additional_features: int. Number of additional trainable dimensions. Only makes sense when
        `partially_freeze=True`. partially_freeze: bool. If True, the regular `weight` will be frozen and extra
        parameters (if any) will be trainable. If False, default to the regular behavior of tf.keras.layers.Dense.
        """
        super().__init__(**kwargs)
        self.out_additional_features = out_additional_features
        self.partially_freeze = partially_freeze

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if out_additional_features > 0:
            self.additional_fc = tf.keras.layers.Dense(
                units=out_additional_features, use_bias=bias, name="additional_fc"
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = tf.linalg.matmul(a=inputs, b=self.weight, transpose_b=True)
        if self.bias is not None:
            output = tf.nn.bias_add(output, self.bias)

        if self.out_additional_features > 0:
            additional_features = self.additional_fc(inputs)
            output = tf.concat([output, additional_features], axis=-1)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "out_features": self.out_features,
                "out_additional_features": self.out_additional_features,
                "bias": self.bias is not None,
                "partially_freeze": self.partially_freeze,
            }
        )
        return config

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, out_additional_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.out_features,
            self.out_additional_features,
            self.bias is not None,
            self.partially_freeze,
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        self.weight = self.add_weight(
            shape=(self.out_features, self.in_features), trainable=not self.partially_freeze, name="weight"
        )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_features,), trainable=not self.partially_freeze, name="bias")
        else:
            self.bias = None
        if getattr(self, "additional_fc", None) is not None:
            with tf.name_scope(self.additional_fc.name):
                self.additional_fc.build(self.in_features)


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


class TFIdeficsRMSNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        """
        TFIdeficsRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        self.weight = self.add_weight(name="weight", shape=[self.hidden_size], initializer="ones")

        super().build(input_shape)

    def call(self, hidden_states):
        variance = tf.math.reduce_mean(tf.math.square(tf.cast(hidden_states, tf.float32)), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [tf.float16, tf.bfloat16]:
            hidden_states = tf.cast(hidden_states, self.weight.dtype)

        return self.weight * hidden_states


class TFIdeficsEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = tf.constant(
            1.0 / (self.base ** (tf.range(start=0, limit=self.dim, delta=2, dtype=tf.float32) / self.dim))
        )

    def _compute_cos_sin(self, seq_len):
        t = tf.range(seq_len, dtype=self.inv_freq.dtype)
        freqs = tf.einsum("i, j -> ij", t, self.inv_freq)  # Outer multiplication
        emb = tf.concat((freqs, freqs), axis=-1)

        return tf.cos(emb), tf.sin(emb)

    def call(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = shape_list(x)[2]
        return self._compute_cos_sin(seq_len=seq_len)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = tf.gather(cos, position_ids)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = tf.gather(sin, position_ids)
    cos = tf.expand_dims(cos, 1)
    sin = tf.expand_dims(sin, 1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TFIdeficsMLP(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gate_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="gate_proj")
        self.down_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, name="down_proj")
        self.up_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="up_proj")
        self.act_fn = get_tf_activation(hidden_act)
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "gate_proj", None) is not None:
            with tf.name_scope(self.gate_proj.name):
                self.gate_proj.build(self.hidden_size)
        if getattr(self, "down_proj", None) is not None:
            with tf.name_scope(self.down_proj.name):
                self.down_proj.build(self.intermediate_size)
        if getattr(self, "up_proj", None) is not None:
            with tf.name_scope(self.up_proj.name):
                self.up_proj.build(self.hidden_size)


class TFIdeficsAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        config: IdeficsConfig = None,
        qk_layer_norms: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.config = config
        self.is_causal = True

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        self.is_cross_attention = is_cross_attention

        self.q_proj = tf.keras.layers.Dense(
            num_heads * self.head_dim,
            use_bias=False,
            name="q_proj",
        )
        self.k_proj = tf.keras.layers.Dense(
            num_heads * self.head_dim,
            use_bias=False,
            name="k_proj",
        )
        self.v_proj = tf.keras.layers.Dense(
            num_heads * self.head_dim,
            use_bias=False,
            name="v_proj",
        )
        self.o_proj = tf.keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="o_proj",
        )
        self.rotary_emb = TFIdeficsEmbedding(self.head_dim, name="rotary_emb")

        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = TFIdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps, name="q_layer_norm")
            self.k_layer_norm = TFIdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps, name="k_layer_norm")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = self.is_cross_attention or key_value_states is not None

        bsz, q_len, _ = shape_list(hidden_states)

        query_states = self._shape(self.q_proj(hidden_states), q_len, bsz)
        if not is_cross_attention:
            key_states = self._shape(self.k_proj(hidden_states), q_len, bsz)
            value_states = self._shape(self.v_proj(hidden_states), q_len, bsz)
        else:
            _, kv_len, _ = shape_list(key_value_states)  # Note that, in this case, `kv_len` == `kv_seq_len`
            key_states = self._shape(self.k_proj(key_value_states), kv_len, bsz)
            value_states = self._shape(self.v_proj(key_value_states), kv_len, bsz)

        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            kv_seq_len += shape_list(past_key_value[0])[-2]
        if not is_cross_attention:
            # Below is to allow symbolic tensors compilation
            if tf.is_tensor(kv_seq_len):
                seq_len = tf.reduce_max(kv_seq_len, q_len)
            else:
                seq_len = max(kv_seq_len, q_len)
            cos, sin = self.rotary_emb(value_states, seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        tf.debugging.assert_equal(
            tf.shape(attention_mask),
            [bsz, 1, q_len, kv_seq_len],
            message=f"Attention weights should be of size {[bsz, 1, q_len, kv_seq_len]}, but is {tf.shape(attention_mask)}",
        )

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        tf.debugging.assert_equal(
            tf.shape(attn_output),
            [bsz, self.num_heads, q_len, self.head_dim],
            message=f"Attention weights should be of size {[bsz, self.num_heads, q_len, self.head_dim]}, but is {tf.shape(attn_output)}",
        )

        attn_output = tf.reshape(tf.transpose(attn_output, perm=[0, 2, 1, 3]), (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        attn_weights = None
        if output_attentions:
            logger.warning_once(
                "attn_weights are not extracted in scaled_dot_product_attention. The model returns None instead"
            )

        return attn_output, attn_weights, past_key_value

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if self.is_cross_attention:
            kv_input_dim = (
                self.hidden_size
                if not hasattr(self.config.vision_config, "embed_dim")
                else self.config.vision_config.embed_dim
            )
        else:
            kv_input_dim = self.hidden_size
        if getattr(self, "o_proj", None) is not None:
            with tf.name_scope(self.o_proj.name):
                self.o_proj.build(self.num_heads * self.head_dim)
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build(self.hidden_size)
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build(kv_input_dim)
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build(kv_input_dim)
        if getattr(self, "rotary_emb", None) is not None:
            with tf.name_scope(self.rotary_emb.name):
                self.rotary_emb.build(None)


class TFIdeficsDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.self_attn = TFIdeficsAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            config=config,
            name="self_attn",
        )
        self.mlp = TFIdeficsMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            name="mlp",
        )
        self.input_layernorm = TFIdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFIdeficsRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, name="post_attention_layernorm"
        )
        self.dropout = config.dropout

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training=False,
    ) -> Tuple[tf.Tensor, Optional[Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(tf.Tensor)`, *optional*): cached past key and value projection states
        """

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
        hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)
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


class TFIdeficsGatedCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.cross_attn = TFIdeficsAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            is_cross_attention=True,
            dropout=config.dropout,
            config=config,
            qk_layer_norms=config.qk_layer_norms,
            name="cross_attn",
        )
        self.mlp = TFIdeficsMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            name="mlp",
        )
        self.input_layernorm = TFIdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFIdeficsRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, name="post_attention_layernorm"
        )
        self.config = config.dropout

        self.act_cross_attn = tf.keras.activations.tanh
        self.act_dense = tf.keras.activations.tanh

        self.alpha_initializer = config.alpha_initializer
        self.alpha_type = config.alpha_type
        self.alphas_initializer_range = config.alphas_initializer_range

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        if self.alpha_initializer == "zeros":
            if self.alpha_type == "vector":
                self.alpha_cross_attn = self.add_weight(
                    shape=(1, 1, self.hidden_size), initializer="zeros", trainable=True, name="alpha_cross_attn"
                )
                self.alpha_dense = self.add_weight(
                    shape=(1, 1, self.hidden_size), initializer="zeros", trainable=True, name="alpha_dense"
                )
            elif self.alpha_type == "float":
                self.alpha_cross_attn = self.add_weight(
                    shape=(1,), initializer="zeros", trainable=True, name="alpha_cross_attn"
                )
                self.alpha_dense = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="alpha_dense")
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({self.alpha_type})")

        elif self.alpha_initializer == "ones":
            if self.alpha_type == "vector":
                self.alpha_cross_attn = self.add_weight(
                    shape=(1, 1, self.hidden_size), initializer="ones", trainable=True, name="alpha_cross_attn"
                )
                self.alpha_dense = self.add_weight(
                    shape=(1, 1, self.hidden_size), initializer="ones", trainable=True, name="alpha_dense"
                )
            elif self.alpha_type == "float":
                self.alpha_cross_attn = self.add_weight(
                    shape=(1,), initializer="ones", trainable=True, name="alpha_cross_attn"
                )
                self.alpha_dense = self.add_weight(shape=(1,), initializer="ones", trainable=True, name="alpha_dense")
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({self.alpha_type})")

        elif self.alpha_initializer in {"normal", "gaussian", "random"}:
            if self.alpha_type == "vector":
                self.alpha_cross_attn = self.add_weight(
                    shape=(1, 1, self.hidden_size),
                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.alphas_initializer_range),
                    trainable=True,
                    name="alpha_cross_attn",
                )
                self.alpha_dense = self.add_weight(
                    shape=(1, 1, self.hidden_size),
                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.alphas_initializer_range),
                    trainable=True,
                    name="alpha_dense",
                )
            elif self.alpha_type == "float":
                self.alpha_cross_attn = self.add_weight(
                    shape=(1,),
                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.alphas_initializer_range),
                    trainable=True,
                    name="alpha_type",
                )
                self.alpha_dense = self.add_weight(
                    shape=(1,),
                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.alphas_initializer_range),
                    trainable=True,
                    name="alpha_dense",
                )
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({self.alpha_type})")

        else:
            raise NotImplementedError(f"Alpha initialization scheme {self.alpha_initializer} not yet implemented!")

        if not (hasattr(self, "alpha_cross_attn") and hasattr(self, "alpha_dense")):
            raise ValueError("Alpha parameters not initialized correctly!")
        with tf.name_scope(self.cross_attn.name):
            self.cross_attn.build(None)
        with tf.name_scope(self.mlp.name):
            self.mlp.build(None)
        with tf.name_scope(self.input_layernorm.name):
            self.input_layernorm.build(None)
        with tf.name_scope(self.post_attention_layernorm.name):
            self.post_attention_layernorm.build(None)
        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        image_hidden_states: Optional[tf.Tensor] = None,
        image_attention_mask: Optional[tf.Tensor] = None,
        cross_attention_gate: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
    ) -> Tuple[tf.Tensor, Optional[Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(tf.Tensor)`, *optional*): cached past key and value projection states
            no_images (`bool`, *optional*, defaults to `False`): If `True` the vision part is ignored
        """
        if image_hidden_states is None:
            raise ValueError(
                "`image_hidden_states` is required for Idefics cross attention module which are visual features to be"
                " conditioned on."
            )

        if cross_attention_gate is None:
            raise ValueError(
                "`cross_attention_gate` is required for Idefics cross attention module to zero-out the cross-attention hidden_states attending to no images."
            )

        if past_key_value is not None:
            raise NotImplementedError("Past key value states are not implemented for Idefics cross attention module.")

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=image_hidden_states,
            attention_mask=image_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = tf.nn.dropout(hidden_states, rate=self.config)
        mask = tf.cast(cross_attention_gate == 0, dtype=hidden_states.dtype)
        # Expand dimensions of mask to match hidden_states
        mask = tf.expand_dims(mask, -1)
        hidden_states = tf.where(
            tf.broadcast_to(mask, tf.shape(hidden_states)) == 1, tf.zeros_like(hidden_states), hidden_states
        )
        # when there are no images the model is used in pure language mode
        # gate = 0 if no_images else 1
        hidden_states = residual + self.act_cross_attn(self.alpha_cross_attn) * hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = tf.nn.dropout(hidden_states, rate=self.config)
        hidden_states = residual + self.act_dense(self.alpha_dense) * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) subclass.
    Use it as a regular TensorFlow Layer and refer to the TensorFlow documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IdeficsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class TFIdeficsPreTrainedModel(TFPreTrainedModel):
    config_class = IdeficsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TFIdeficsDecoderLayer", "TFIdeficsGatedCrossAttentionLayer"]


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
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
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
@keras_serializable
class TFIdeficsMainLayer(tf.keras.layers.Layer):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    config_class = IdeficsConfig

    def __init__(self, config: IdeficsConfig, add_pooling_year: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = TFIdeficsDecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            name="embed_tokens",
        )

        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        self.vision_model = TFIdeficsVisionTransformer(config.vision_config, name="vision_model")

        # Perceiver Resampler
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = TFIdeficsPerceiverResampler(
                config,
                config.vision_config.embed_dim,
                perceiver_config.resampler_depth,
                perceiver_config.resampler_n_heads,
                perceiver_config.resampler_head_dim,
                perceiver_config.resampler_n_latents,
                name="perceiver_resampler",
            )

        self.decoder_layers = [
            TFIdeficsDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = [
            TFIdeficsGatedCrossAttentionLayer(config, name=f"gated_cross_attn_layers.{i}")
            for i in range(num_cross_layers)
        ]
        self.gradient_checkpointing = False

        self.norm = TFIdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="norm")

        self.gradient_checkpointing = False
        self.freeze_relevant_params(config)

    def freeze_relevant_params(self, config=None):
        if config is None:
            config = self.config

        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)

        if config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    def freeze_text_layers(self, module_exceptions=[]):
        for module in [self.decoder_layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    def freeze_vision_layers(self, module_exceptions=[]):
        freeze_model(self.vision_model, module_exceptions=module_exceptions)

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
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        pixel_values: Optional[tf.Tensor] = None,
        image_encoder_embeddings: Optional[tf.Tensor] = None,
        perceiver_embeddings: Optional[tf.Tensor] = None,
        image_attention_mask: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[TFIdeficsBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = tf.math.cumsum(tf.cast(attention_mask, dtype=tf.int32), axis=-1) - 1
            position_ids = tf.where(attention_mask == 0, 1, position_ids)
        elif position_ids is None:
            position_ids = tf.range(past_key_values_length, seq_length + past_key_values_length, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, 0)

        no_images = False
        if (
            sum((int(pixel_values is None), int(image_encoder_embeddings is None), int(perceiver_embeddings is None)))
            != 2
        ):
            raise ValueError(
                "Exactly 1 of pixel_values, image_encoder_embeddings or perceiver_embeddings has to be not-None."
            )

        elif pixel_values is not None:
            no_images = tf.reduce_sum(tf.cast(pixel_values, dtype=tf.int32)) == 0
            pixel_values = tf.cast(pixel_values, dtype=self.dtype)  # fp16 compatibility
            # Below hack is because when cross-loading pytorch weights, there is an
            # initial forward pass with dummy input and code below is here to handle that
            if len(pixel_values.shape) == 4:
                batch_size = shape_list(pixel_values)[0]
                num_images = shape_list(pixel_values)[0]
                # pixel_values = tf.reshape(pixel_values, [batch_size * num_images, *pixel_values.shape[1:]])
            elif len(pixel_values.shape) == 5:
                batch_size, num_images = shape_list(pixel_values)[:2]
                pixel_values = tf.reshape(pixel_values, [batch_size * num_images, *pixel_values.shape[2:]])

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
            ).last_hidden_state

        elif image_encoder_embeddings is not None:
            batch_size, num_images, image_seq_len, image_hidden_size = shape_list(image_encoder_embeddings)
            image_hidden_states = tf.cast(image_encoder_embeddings, dtype=self.dtype)
            image_hidden_states = tf.reshape(
                image_hidden_states, (batch_size * num_images, image_seq_len, image_hidden_size)
            )

        if self.config.use_resampler:
            if perceiver_embeddings is None:
                perceiver_embeddings = self.perceiver_resampler(image_hidden_states)
                image_seq_len, image_hidden_size = shape_list(perceiver_embeddings)[1:3]
            else:
                batch_size, num_images, image_seq_len, image_hidden_size = shape_list(perceiver_embeddings)
            image_hidden_states = perceiver_embeddings
        elif perceiver_embeddings is None:
            image_seq_len, image_hidden_size = shape_list(image_hidden_states)[1:3]
        else:
            raise ValueError("If `perceiver_embeddings` are passed, use_resampler should be True")

        image_hidden_states = tf.reshape(
            image_hidden_states, (batch_size, num_images * image_seq_len, image_hidden_size)
        )
        # # Hack to use the model in full language modeling mode
        # image_attention_mask = tf.zeros((batch_size, seq_length, 1), dtype=tf.int32)

        # this is to account for the dummy inputs
        if pixel_values is not None and len(pixel_values.shape) == 4 and image_attention_mask is None:
            image_attention_mask = tf.zeros((batch_size, seq_length, 1), dtype=tf.int32)

        text_seq_len = shape_list(image_attention_mask)[1]
        image_attention_mask = tf.expand_dims(image_attention_mask, -1)
        image_attention_mask = tf.repeat(image_attention_mask, repeats=image_seq_len)
        image_attention_mask = tf.reshape(image_attention_mask, (batch_size, text_seq_len, num_images * image_seq_len))

        if image_hidden_states is not None:
            image_batch_size, image_sequence_length, _ = shape_list(image_hidden_states)
            image_hidden_shape = (image_batch_size, image_sequence_length)
            if image_attention_mask is None:
                image_attention_mask = tf.ones(image_hidden_shape, dtype=tf.int32)
            image_attention_mask = invert_attention_mask(image_attention_mask)
        else:
            image_attention_mask = None

        cross_attention_gate = tf.squeeze(
            tf.cast(tf.reduce_any(image_attention_mask == 0, axis=-1), dtype=self.dtype), axis=1
        )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = tf.ones((batch_size, seq_length_with_past), dtype=tf.bool)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            def vblock(
                main_block,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                image_hidden_states,
                image_attention_mask,
                cross_attention_gate,
                output_attentions,
                use_cache,
                layer_idx,
                cross_layer_interval,
                gated_cross_attn_layers,
            ):
                # TODO(ls): Add cross attention values to respective lists
                if layer_idx % cross_layer_interval == 0:
                    xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
                    outputs = xblock(
                        hidden_states,
                        attention_mask=attention_mask,
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        cross_attention_gate=cross_attention_gate,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        past_key_value=None,  # not implemented
                    )
                    hidden_states = outputs[0]

                layer_outputs = main_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                return layer_outputs

            if self.gradient_checkpointing and training:
                past_key_value = None
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                layer_outputs = tf.recompute_grad(
                    vblock,
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    image_hidden_states,
                    image_attention_mask,
                    output_attentions,
                    use_cache,
                    no_images,
                    idx,
                    self.cross_layer_interval,
                    self.gated_cross_attn_layers,
                )
            else:
                layer_outputs = vblock(
                    decoder_layer,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    image_hidden_states=image_hidden_states,
                    image_attention_mask=image_attention_mask,
                    cross_attention_gate=cross_attention_gate,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    layer_idx=idx,
                    cross_layer_interval=self.cross_layer_interval,
                    gated_cross_attn_layers=self.gated_cross_attn_layers,
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
        image_hidden_states = tf.reshape(
            image_hidden_states, (batch_size, num_images, image_seq_len, image_hidden_size)
        )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, image_hidden_states]
                if v is not None
            )
        return TFIdeficsBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build(None)
        if getattr(self, "perceiver_resampler", None) is not None:
            with tf.name_scope(self.perceiver_resampler.name):
                self.perceiver_resampler.build(None)
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, "gated_cross_attn_layers", None) is not None:
            for layer in self.gated_cross_attn_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFIdeficsModel(TFIdeficsPreTrainedModel):
    def __init__(self, config: IdeficsConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.model = TFIdeficsMainLayer(config, name="model")

    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        pixel_values: Optional[tf.Tensor] = None,
        image_encoder_embeddings: Optional[tf.Tensor] = None,
        perceiver_embeddings: Optional[tf.Tensor] = None,
        image_attention_mask: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[TFIdeficsBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)


class TFIdeficsForVisionText2Text(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    config_class = IdeficsConfig

    def __init__(self, config, vision_model=None, **kwargs):
        super().__init__(config, **kwargs)
        self.model = TFIdeficsMainLayer(config, name="model")
        self.lm_head = TFIdeficsDecoupledLinear(
            config.hidden_size,
            config.vocab_size,
            config.additional_vocab_size,
            bias=False,
            partially_freeze=config.freeze_lm_head,
            name="lm_head",
        )

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

    def tie_weights(self):
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of
        IdeficsDecoupledLinear and IdeficsDecoupledEmbedding.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()

        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings.weight = input_embeddings.weight
            if input_embeddings.num_additional_embeddings > 0:
                assert output_embeddings.out_additional_features == input_embeddings.num_additional_embeddings
                output_embeddings.additional_fc.weight = input_embeddings.additional_embedding.weight

        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
            if hasattr(output_embeddings, "out_additional_features") and hasattr(
                input_embeddings, "num_additional_embeddings"
            ):
                output_embeddings.out_additional_features = input_embeddings.num_additional_embeddings

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFIdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        pixel_values: Optional[tf.Tensor] = None,
        image_encoder_embeddings: Optional[tf.Tensor] = None,
        perceiver_embeddings: Optional[tf.Tensor] = None,
        image_attention_mask: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        training=False,
    ) -> Union[TFIdeficsCausalLMOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        Args:
            labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >> from transformers import AutoTokenizer, TFIdeficsForVisionText2Text

        >> model = TFIdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
        >> tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/idefics-9b")

        >> prompt = "Hey, are you consciours? Can you talk to me?"
        >> inputs = tokenizer(prompt, return_tensors="tf")

        >> # Generate
        >> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0]
                shift_labels = labels[..., 1:][shift_attention_mask != 0]
            else:
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = self.hf_compute_loss(
                labels=tf.reshape(shift_labels, [-1]), logits=tf.reshape(shift_logits, [-1, shift_logits.shape[-1]])
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TFIdeficsCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        image_hidden_states = kwargs.pop("image_hidden_states", None)
        if image_hidden_states is not None:
            if self.config.use_resampler:
                kwargs["perceiver_embeddings"] = image_hidden_states
            else:
                kwargs["image_encoder_embeddings"] = image_hidden_states
            kwargs["pixel_values"] = None
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        unwanted_kwargs = ["token_type_ids"]
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        *args,
        **model_kwargs,
    ):
        return expand_inputs_for_generation(*args, **model_kwargs)

    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder):
        return update_model_kwargs_for_generation(outputs, model_kwargs)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(tf.gather(past_state, beam_idx) for past_state in layer_past),)
        return reordered_past

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)


__all__ = ["TFIdeficsForVisionText2Text", "TFIdeficsModel", "TFIdeficsPreTrainedModel"]
