# coding=utf-8
# Copyright 2022 The EleutherAI and HuggingFace Teams. All rights reserved.
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
"""TF 2.0 GPT-J model."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSharedEmbeddings,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-j-6B"
_CONFIG_FOR_DOC = "GPTJConfig"


def create_sinusoidal_positions(num_pos: int, dim: int) -> tf.Tensor:
    inv_freq = tf.cast(1.0 / (10000 ** (tf.range(0, dim, 2) / dim)), tf.float32)
    sinusoid_inp = tf.cast(tf.einsum("i , j -> i j", tf.range(num_pos, dtype=tf.float32), inv_freq), tf.float32)
    sin, cos = tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)
    out = tf.concat((sin, cos), axis=1)
    return out


def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    rotate_half_tensor = tf.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), axis=-1)
    new_shape = shape_list(rotate_half_tensor)[:-2] + [tf.math.reduce_prod(shape_list(rotate_half_tensor)[-2:])]
    rotate_half_tensor = tf.reshape(rotate_half_tensor, new_shape)
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor: tf.Tensor, sincos: tf.Tensor) -> tf.Tensor:
    sin_pos, cos_pos = sincos
    sin_pos = tf.repeat(sin_pos[:, :, None, :], 2, 3)
    cos_pos = tf.repeat(cos_pos[:, :, None, :], 2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class TFGPTJAttention(keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = self.head_dim**0.5
        self.rotary_dim = config.rotary_dim

        self.attn_dropout = keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = keras.layers.Dropout(config.resid_pdrop)

        self.q_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="v_proj",
        )
        self.out_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj",
        )

        self.max_positions = config.max_position_embeddings
        self.lower_triangle_mask = tf.reshape(
            tf.cast(tf.experimental.numpy.tril(tf.ones((self.max_positions, self.max_positions))), tf.int8),
            (1, 1, self.max_positions, self.max_positions),
        )
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(self.max_positions, pos_embd_dim)

    def get_causal_mask(self, key_length, query_length) -> tf.Tensor:
        return tf.cast(self.lower_triangle_mask[:, :, key_length - query_length : key_length, :key_length], tf.bool)

    @staticmethod
    def get_masked_bias(dtype: tf.DType) -> tf.Tensor:
        return tf.cast(tf.constant(-1e9), dtype)

    def _split_heads(self, hidden_states: tf.Tensor, rotary: bool) -> tf.Tensor:
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = shape_list(hidden_states)[:-1] + [self.num_attention_heads, self.head_dim]
        hidden_states = tf.reshape(hidden_states, new_shape)
        if rotary:
            return hidden_states
        if len(shape_list(hidden_states)) == 4:
            return tf.transpose(hidden_states, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)
        if len(shape_list(hidden_states)) == 5:
            return tf.transpose(hidden_states, (0, 1, 3, 2, 4))  # (batch, blocks, head, block_length, head_features)
        raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}")

    def _merge_heads(self, hidden_states: tf.Tensor) -> tf.Tensor:
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(shape_list(hidden_states)) == 4:
            hidden_states = tf.transpose(hidden_states, (0, 2, 1, 3))
        elif len(shape_list(hidden_states)) == 5:
            hidden_states = tf.transpose(hidden_states, (0, 1, 3, 2, 4))
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}")
        new_shape = shape_list(hidden_states)[:-2] + [self.num_attention_heads * self.head_dim]
        return tf.reshape(hidden_states, new_shape)

    def _attn(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # compute causal mask from causal mask buffer
        query_length, key_length = shape_list(query)[-2], shape_list(key)[-2]
        causal_mask = self.get_causal_mask(key_length, query_length)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

        attn_weights = tf.matmul(query, key, transpose_b=True)
        attn_weights = tf.where(causal_mask, attn_weights, self.get_masked_bias(attn_weights.dtype))

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = tf.matmul(attn_weights, value)

        return attn_output, attn_weights

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, True)
        key = self._split_heads(key, True)
        value = self._split_heads(value, False)

        sincos = tf.cast(tf.gather(self.embed_positions, position_ids, axis=0), hidden_states.dtype)
        sincos = tf.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = tf.concat((k_rot, k_pass), axis=-1)
            query = tf.concat((q_rot, q_pass), axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        key = tf.transpose(key, (0, 2, 1, 3))
        query = tf.transpose(query, (0, 2, 1, 3))

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])


class TFGPTJMLP(keras.layers.Layer):
    def __init__(self, intermediate_size: int, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        embed_dim = config.n_embd

        self.fc_in = keras.layers.Dense(
            intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="fc_in"
        )
        self.fc_out = keras.layers.Dense(
            embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="fc_out"
        )

        self.act = get_tf_activation(config.activation_function)
        self.dropout = keras.layers.Dropout(config.embd_pdrop)
        self.embed_dim = config.n_embd
        self.intermediate_size = intermediate_size

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "fc_in", None) is not None:
            with tf.name_scope(self.fc_in.name):
                self.fc_in.build([None, None, self.embed_dim])
        if getattr(self, "fc_out", None) is not None:
            with tf.name_scope(self.fc_out.name):
                self.fc_out.build([None, None, self.intermediate_size])


class TFGPTJBlock(keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = TFGPTJAttention(config, name="attn")
        self.mlp = TFGPTJMLP(inner_dim, config, name="mlp")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )  # attn_outputs: attn_output, present, (attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs  # hidden_states, present, (attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "ln_1", None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.config.n_embd])
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)


@keras_serializable
class TFGPTJMainLayer(keras.layers.Layer):
    config_class = GPTJConfig

    def __init__(self, config: GPTJConfig, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        self.wte = TFSharedEmbeddings(
            config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name="wte"
        )
        self.drop = keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFGPTJBlock(config, name=f"h_._{i}") for i in range(config.n_layer)]
        self.ln_f = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        self.embed_dim = config.n_embd

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value: tf.Tensor):
        self.wte.weight = value
        self.wte.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = shape_list(past_key_values[0][0])[-2]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_length, input_shape[-1] + past_length), axis=0)

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask_shape = shape_list(attention_mask)
            attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(tf.subtract(one_cst, attention_mask), tf.constant(-10000.0))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.wte.vocab_size)
            inputs_embeds = self.wte(input_ids, mode="embedding")

        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.wte(token_type_ids, mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                training=training,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "wte", None) is not None:
            with tf.name_scope(self.wte.name):
                self.wte.build(None)
        if getattr(self, "ln_f", None) is not None:
            with tf.name_scope(self.ln_f.name):
                self.ln_f.build([None, None, self.embed_dim])
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFGPTJPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTJConfig
    base_model_prefix = "transformer"
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias"]


GPTJ_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

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
        config ([`GPTJConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPTJ_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past` is `None` else `past[0].shape[-2]` (`sequence_length` of
            input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past` output below). Can be used to speed up sequential decoding. The token ids which have their past
            given to this model should not be passed as input ids as they have already been computed.
        attention_mask (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.",
    GPTJ_START_DOCSTRING,
)
class TFGPTJModel(TFGPTJPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPTJMainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """

        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)


@add_start_docstrings(
    """
    The GPT-J Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForCausalLM(TFGPTJPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPTJMainLayer(config, name="transformer")
        self.lm_head = keras.layers.Dense(
            config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name="lm_head"
        )
        self.config = config

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFCausalLMOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # shift labels to the left and cut last logit token
            shifted_logits = lm_logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.n_embd])


@add_start_docstrings(
    """
    The GPT-J Model transformer with a sequence classification head on top (linear layer).

    [`GPTJForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT, GPT-2, GPT-Neo) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForSequenceClassification(TFGPTJPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFGPTJMainLayer(config, name="transformer")
        self.score = keras.layers.Dense(
            self.num_labels,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if labels is not None and self.config.pad_token_id is None and input_ids.shape[0] != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
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
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "score", None) is not None:
            with tf.name_scope(self.score.name):
                self.score.build([None, None, self.config.n_embd])


@add_start_docstrings(
    """
    The GPT-J Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForQuestionAnswering(TFGPTJPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFGPTJMainLayer(config, name="transformer")
        self.qa_outputs = keras.layers.Dense(
            self.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])


__all__ = [
    "TFGPTJForCausalLM",
    "TFGPTJForQuestionAnswering",
    "TFGPTJForSequenceClassification",
    "TFGPTJModel",
    "TFGPTJPreTrainedModel",
]
