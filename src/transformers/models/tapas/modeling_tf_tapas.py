# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team.
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
"""TF 2.0 TAPAS model."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    logging,
    replace_return_docstrings,
)
from .configuration_tapas import TapasConfig


logger = logging.get_logger(__name__)

# soft dependency
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp

        # On the first call, check whether a compatible version of TensorFlow is installed
        # TensorFlow Probability depends on a recent stable release of TensorFlow
        n = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        logger.error(
            "TAPAS models are not usable since `tensorflow_probability` can't be loaded. "
            "It seems you have `tensorflow_probability` installed with the wrong tensorflow version. "
            "Please try to reinstall it following the instructions here: https://github.com/tensorflow/probability."
        )
else:
    try:
        import tensorflow_probability as tfp

        # On the first call, check whether a compatible version of TensorFlow is installed
        # TensorFlow Probability depends on a recent stable release of TensorFlow
        _ = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        pass

_CONFIG_FOR_DOC = "TapasConfig"
_CHECKPOINT_FOR_DOC = "google/tapas-base"


EPSILON_ZERO_DIVISION = 1e-10
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


@dataclass
class TFTableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`TFTapasForQuestionAnswering`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (`tf.Tensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    logits_aggregation: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


class TFTapasEmbeddings(keras.layers.Layer):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)
        self.reset_position_index_per_cell = config.reset_position_index_per_cell
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        for i, type_vocab_size in enumerate(self.config.type_vocab_sizes):
            with tf.name_scope(f"token_type_embeddings_{i}"):
                setattr(
                    self,
                    f"token_type_embeddings_{i}",
                    self.add_weight(
                        name="embeddings",
                        shape=[type_vocab_size, self.hidden_size],
                        initializer=get_initializer(self.initializer_range),
                    ),
                )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape + [self.number_of_token_type_embeddings], value=0)

        if position_ids is None:
            # create absolute position embeddings
            position_ids = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
            position_ids = tf.broadcast_to(position_ids, shape=input_shape)
            # when self.config.reset_position_index_per_cell is set to True, create relative position embeddings
            if self.reset_position_index_per_cell:
                # shape (batch_size, seq_len)
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                # shape (batch_size, seq_len)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                # shape (batch_size, seq_len)
                full_index = ProductIndexMap(col_index, row_index)
                # shape (max_rows * max_columns,). First absolute position for every cell
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                # ? shape (batch_size, seq_len). First absolute position of the cell for every token
                first_position = gather(first_position_per_segment, full_index)
                # shape (1, seq_len)
                position = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
                position_ids = tf.math.minimum(self.max_position_embeddings - 1, position - first_position)

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        position_embeddings = tf.gather(self.position_embeddings, indices=position_ids)

        final_embeddings = inputs_embeds + position_embeddings

        for i in range(self.number_of_token_type_embeddings):
            name = f"token_type_embeddings_{i}"
            final_embeddings += tf.gather(params=getattr(self, name), indices=token_type_ids[:, :, i])

        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->Tapas
class TFTapasSelfAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(inputs=encoder_hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=encoder_hidden_states), batch_size)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFTapasModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->Tapas
class TFTapasSelfOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->Tapas
class TFTapasAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFTapasSelfAttention(config, name="self")
        self.dense_output = TFTapasSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # add attentions (possibly with past_key_value) if we output them
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->Tapas
class TFTapasIntermediate(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->Tapas
class TFTapasOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->Tapas
class TFTapasLayer(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFTapasAttention(config, name="attention")
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = TFTapasAttention(config, name="crossattention")
        self.intermediate = TFTapasIntermediate(config, name="intermediate")
        self.bert_output = TFTapasOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)


# Copied from transformers.models.bert.modeling_tf_bert.TFBertEncoder with Bert->Tapas
class TFTapasEncoder(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFTapasLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->Tapas
class TFTapasPooler(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform with Bert->Tapas
class TFTapasPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead with Bert->Tapas
class TFTapasLMPredictionHead(keras.layers.Layer):
    def __init__(self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size

        self.transform = TFTapasPredictionHeadTransform(config, name="transform")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


# Copied from transformers.models.bert.modeling_tf_bert.TFBertMLMHead with Bert->Tapas
class TFTapasMLMHead(keras.layers.Layer):
    def __init__(self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TFTapasLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


@keras_serializable
class TFTapasMainLayer(keras.layers.Layer):
    config_class = TapasConfig

    def __init__(self, config: TapasConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFTapasEmbeddings(config, name="embeddings")
        self.encoder = TFTapasEncoder(config, name="encoder")
        self.pooler = TFTapasPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape + [len(self.config.type_vocab_sizes)], value=0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            training=training,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFTapasPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"

    @property
    def input_signature(self):
        return {
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None), tf.float32, name="attention_mask"),
            "token_type_ids": tf.TensorSpec((None, None, 7), tf.int32, name="token_type_ids"),
        }


TAPAS_START_DOCSTRING = r"""

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
        config ([`TapasConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TAPAS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0}, 7)`, *optional*):
            Token indices that encode tabular structure. Indices can be obtained using [`AutoTokenizer`]. See this
            class for more info.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. If
            `reset_position_index_per_cell` of [`TapasConfig`] is set to `True`, relative position embeddings will be
            used. Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.",
    TAPAS_START_DOCSTRING,
)
class TFTapasModel(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.tapas = TFTapasMainLayer(config, name="tapas")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasModel
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasModel.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        outputs = self.tapas(
            input_ids=input_ids,
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

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)


@add_start_docstrings("""Tapas Model with a `language modeling` head on top.""", TAPAS_START_DOCSTRING)
class TFTapasForMaskedLM(TFTapasPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFTapasForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.tapas = TFTapasMainLayer(config, add_pooling_layer=False, name="tapas")
        self.lm_head = TFTapasMLMHead(config, input_embeddings=self.tapas.embeddings, name="cls")

    def get_lm_head(self) -> keras.layers.Layer:
        return self.lm_head.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForMaskedLM
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)

        >>> inputs = tokenizer(
        ...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="tf"
        ... )
        >>> labels = tokenizer(
        ...     table=table, queries="How many movies has George Clooney played in?", return_tensors="tf"
        ... )["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> logits = outputs.logits
        ```"""
        outputs = self.tapas(
            input_ids=input_ids,
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
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)


class TFTapasComputeTokenLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        self.temperature = config.temperature
        # cell selection heads
        with tf.name_scope("output"):
            self.output_weights = self.add_weight(
                name="output_weights",
                shape=(config.hidden_size,),
                dtype=tf.float32,
                trainable=True,
                initializer=tf.zeros_initializer()
                if config.init_cell_selection_weights_to_zero
                else keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            )
            self.output_bias = self.add_weight(
                name="output_bias", shape=(), trainable=True, initializer=tf.zeros_initializer()
            )

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        """
        Computes logits per token

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.

        Returns:
            logits (`tf.Tensor` of shape `(batch_size, sequence_length)`): Logits per token.
        """
        logits = (tf.einsum("bsj,j->bs", sequence_output, self.output_weights) + self.output_bias) / self.temperature
        return logits


class TFTapasComputeColumnLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)

        with tf.name_scope("column_output"):
            self.column_output_weights = self.add_weight(
                name="column_output_weights",
                shape=[config.hidden_size],
                dtype=tf.float32,
                trainable=True,
                initializer=tf.zeros_initializer()
                if config.init_cell_selection_weights_to_zero
                else keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            )
            self.column_output_bias = self.add_weight(
                name="column_output_bias", shape=(), trainable=True, initializer=tf.zeros_initializer()
            )

    def call(self, sequence_output, cell_index, cell_mask, allow_empty_column_selection) -> tf.Tensor:
        """
        Computes the column logits.

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.
            cell_index (`ProductIndexMap`):
                Index that groups tokens into cells.
            cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
                Mask for cells that exist in the table (i.e. that are not padding).
            allow_empty_column_selection (`bool`):
                Whether to allow not to select any column

        Returns:
            column_logits (`tf.Tensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits for
            every example in the batch.
        """

        # First, compute the token logits (batch_size, seq_len) - without temperature
        token_logits = tf.einsum("bsj,j->bs", sequence_output, self.column_output_weights) + self.column_output_bias

        # Next, average the logits per cell (batch_size, max_num_cols*max_num_rows)
        cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

        # Finally, average the logits per column (batch_size, max_num_cols)
        column_index = cell_index.project_inner(cell_logits_index)
        column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

        cell_count, _ = reduce_sum(cell_mask, column_index)
        column_logits /= cell_count + EPSILON_ZERO_DIVISION

        # Mask columns that do not appear in the example.
        is_padding = tf.logical_and(cell_count < 0.5, tf.not_equal(out_index.indices, 0))
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(is_padding, tf.float32)

        if not allow_empty_column_selection:
            column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(tf.equal(out_index.indices, 0), tf.float32)

        return column_logits


@add_start_docstrings(
    """
    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
    SQA, WTQ or WikiSQL-supervised tasks.
    """,
    TAPAS_START_DOCSTRING,
)
class TFTapasForQuestionAnswering(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # base model
        self.tapas = TFTapasMainLayer(config, name="tapas")

        # dropout
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)

        self.compute_token_logits = TFTapasComputeTokenLogits(config, name="compute_token_logits")

        self.compute_column_logits = TFTapasComputeColumnLogits(config, name="compute_column_logits")

        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = keras.layers.Dense(
                config.num_aggregation_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="aggregation_classifier",
            )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFTableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        table_mask: np.ndarray | tf.Tensor | None = None,
        aggregation_labels: np.ndarray | tf.Tensor | None = None,
        float_answer: np.ndarray | tf.Tensor | None = None,
        numeric_values: np.ndarray | tf.Tensor | None = None,
        numeric_values_scale: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFTableQuestionAnsweringOutput, Tuple[tf.Tensor]]:
        r"""
        table_mask (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and
            padding are 0.
        labels (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Labels per token for computing the hierarchical cell selection loss. This encodes the positions of the
            answer appearing in the table. Can be obtained using [`AutoTokenizer`].

            - 1 for tokens that are **part of the answer**,
            - 0 for tokens that are **not part of the answer**.

        aggregation_labels (`tf.Tensor` of shape `(batch_size, )`, *optional*):
            Aggregation function index for every example in the batch for computing the aggregation loss. Indices
            should be in `[0, ..., config.num_aggregation_labels - 1]`. Only required in case of strong supervision for
            aggregation (WikiSQL-supervised).
        float_answer (`tf.Tensor` of shape `(batch_size, )`, *optional*):
            Float answer for every example in the batch. Set to *float('nan')* for cell selection questions. Only
            required in case of weak supervision (WTQ) to calculate the aggregate mask and regression loss.
        numeric_values (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Numeric values of every token, NaN for tokens which are not numeric values. Can be obtained using
            [`AutoTokenizer`]. Only required in case of weak supervision for aggregation (WTQ) to calculate the
            regression loss.
        numeric_values_scale (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Scale of the numeric values of every token. Can be obtained using [`AutoTokenizer`]. Only required in case
            of weak supervision for aggregation (WTQ) to calculate the regression loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForQuestionAnswering
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        >>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> logits_aggregation = outputs.logits_aggregation
        ```"""

        outputs = self.tapas(
            input_ids=input_ids,
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

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)

        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        # Construct indices for the table.
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape + [len(self.config.type_vocab_sizes)], 0)

        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        row_ids = token_type_ids[:, :, token_types.index("row_ids")]
        column_ids = token_type_ids[:, :, token_types.index("column_ids")]

        # Construct indices for the table.
        row_index = IndexMap(
            indices=tf.minimum(tf.cast(row_ids, tf.int32), self.config.max_num_rows - 1),
            num_segments=self.config.max_num_rows,
            batch_dims=1,
        )
        col_index = IndexMap(
            indices=tf.minimum(tf.cast(column_ids, tf.int32), self.config.max_num_columns - 1),
            num_segments=self.config.max_num_columns,
            batch_dims=1,
        )
        cell_index = ProductIndexMap(row_index, col_index)

        # Masks.
        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)[:-1]
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)
        # Table cells only, without question tokens and table headers.
        if table_mask is None:
            table_mask = tf.where(row_ids > 0, tf.ones_like(row_ids), tf.zeros_like(row_ids))
        # <float32>[batch_size, seq_length]
        input_mask_float = tf.cast(attention_mask, tf.float32)
        table_mask_float = tf.cast(table_mask, tf.float32)

        # Mask for cells that exist in the table (i.e. that are not padding).
        cell_mask, _ = reduce_mean(input_mask_float, cell_index)

        # Compute logits per token. These are used to select individual cells.
        logits = self.compute_token_logits(sequence_output)

        # Compute logits per column. These are used to select a column.
        column_logits = None
        if self.config.select_one_column:
            column_logits = self.compute_column_logits(
                sequence_output, cell_index, cell_mask, self.config.allow_empty_column_selection
            )

        # Aggregate logits.
        logits_aggregation = None
        if self.config.num_aggregation_labels > 0:
            logits_aggregation = self.aggregation_classifier(pooled_output)

        # Total loss calculation
        total_loss = tf.zeros(shape=(1,), dtype=tf.float32)
        calculate_loss = False
        if labels is not None:
            calculate_loss = True
            is_supervised = not self.config.num_aggregation_labels > 0 or not self.config.use_answer_as_supervision

            # Semi-supervised cell selection in case of no aggregation:
            # If the answer (the denotation) appears directly in the table we might
            # select the answer without applying any aggregation function. There are
            # some ambiguous cases, see utils._calculate_aggregate_mask for more info.
            # `aggregate_mask` is 1 for examples where we chose to aggregate and 0
            #  for examples where we chose to select the answer directly.
            # `labels` encodes the positions of the answer appearing in the table.
            if is_supervised:
                aggregate_mask = None
            else:
                if float_answer is not None:
                    assert (
                        shape_list(labels)[0] == shape_list(float_answer)[0]
                    ), "Make sure the answers are a FloatTensor of shape (batch_size,)"
                    # <float32>[batch_size]
                    aggregate_mask = _calculate_aggregate_mask(
                        float_answer,
                        pooled_output,
                        self.config.cell_selection_preference,
                        labels,
                        self.aggregation_classifier,
                    )
                else:
                    aggregate_mask = None
                    raise ValueError("You have to specify float answers in order to calculate the aggregate mask")

            # Cell selection log-likelihood
            if self.config.average_logits_per_cell:
                logits_per_cell, _ = reduce_mean(logits, cell_index)
                logits = gather(logits_per_cell, cell_index)
            dist_per_token = tfp.distributions.Bernoulli(logits=logits)

            # Compute cell selection loss per example.
            selection_loss_per_example = None
            if not self.config.select_one_column:
                weight = tf.where(
                    labels == 0,
                    tf.ones_like(labels, dtype=tf.float32),
                    self.config.positive_label_weight * tf.ones_like(labels, dtype=tf.float32),
                )
                selection_loss_per_token = -dist_per_token.log_prob(labels) * weight
                selection_loss_per_example = tf.reduce_sum(selection_loss_per_token * input_mask_float, axis=1) / (
                    tf.reduce_sum(input_mask_float, axis=1) + EPSILON_ZERO_DIVISION
                )
            else:
                selection_loss_per_example, logits = _single_column_cell_selection_loss(
                    logits, column_logits, labels, cell_index, col_index, cell_mask
                )
                dist_per_token = tfp.distributions.Bernoulli(logits=logits)

            # Supervised cell selection
            if self.config.disable_per_token_loss:
                pass
            elif is_supervised:
                total_loss += tf.reduce_mean(selection_loss_per_example)
            else:
                # For the not supervised case, do not assign loss for cell selection
                total_loss += tf.reduce_mean(selection_loss_per_example * (1.0 - aggregate_mask))

            # Semi-supervised regression loss and supervised loss for aggregations
            if self.config.num_aggregation_labels > 0:
                if is_supervised:
                    # Note that `aggregate_mask` is None if the setting is supervised.
                    if aggregation_labels is not None:
                        assert (
                            shape_list(labels)[0] == shape_list(aggregation_labels)[0]
                        ), "Make sure the aggregation labels are a LongTensor of shape (batch_size,)"
                        per_example_additional_loss = _calculate_aggregation_loss(
                            logits_aggregation,
                            aggregate_mask,
                            aggregation_labels,
                            self.config.use_answer_as_supervision,
                            self.config.num_aggregation_labels,
                            self.config.aggregation_loss_weight,
                        )
                    else:
                        raise ValueError(
                            "You have to specify aggregation labels in order to calculate the aggregation loss"
                        )
                else:
                    aggregation_labels = tf.zeros(shape_list(labels)[0], dtype=tf.int32)
                    per_example_additional_loss = _calculate_aggregation_loss(
                        logits_aggregation,
                        aggregate_mask,
                        aggregation_labels,
                        self.config.use_answer_as_supervision,
                        self.config.num_aggregation_labels,
                        self.config.aggregation_loss_weight,
                    )

                if self.config.use_answer_as_supervision:
                    if numeric_values is not None and numeric_values_scale is not None:
                        assert shape_list(numeric_values) == shape_list(numeric_values_scale)
                        # Add regression loss for numeric answers which require aggregation.
                        answer_loss, large_answer_loss_mask = _calculate_regression_loss(
                            float_answer,
                            aggregate_mask,
                            dist_per_token,
                            numeric_values,
                            numeric_values_scale,
                            table_mask_float,
                            logits_aggregation,
                            self.config,
                        )
                        per_example_additional_loss += answer_loss
                        # Zero loss for examples with answer_loss > cutoff.
                        per_example_additional_loss *= large_answer_loss_mask
                    else:
                        raise ValueError(
                            "You have to specify numeric values and numeric values scale in order to calculate the"
                            " regression loss"
                        )
                total_loss += tf.reduce_mean(per_example_additional_loss)

        else:
            # if no label ids are provided, set them to zeros in order to properly compute logits
            labels = tf.zeros_like(logits)
            _, logits = _single_column_cell_selection_loss(
                logits, column_logits, labels, cell_index, col_index, cell_mask
            )
        if not return_dict:
            output = (logits, logits_aggregation) + outputs[2:]
            return ((total_loss,) + output) if calculate_loss else output

        return TFTableQuestionAnsweringOutput(
            loss=total_loss if calculate_loss else None,
            logits=logits,
            logits_aggregation=logits_aggregation,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        if getattr(self, "compute_token_logits", None) is not None:
            with tf.name_scope(self.compute_token_logits.name):
                self.compute_token_logits.build(None)
        if getattr(self, "compute_column_logits", None) is not None:
            with tf.name_scope(self.compute_column_logits.name):
                self.compute_column_logits.build(None)
        if getattr(self, "aggregation_classifier", None) is not None:
            with tf.name_scope(self.aggregation_classifier.name):
                self.aggregation_classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
    entailment tasks, such as TabFact (Chen et al., 2020).
    """,
    TAPAS_START_DOCSTRING,
)
class TFTapasForSequenceClassification(TFTapasPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.tapas = TFTapasMainLayer(config, name="tapas")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
            "classification_class_index" in the original implementation.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForSequenceClassification
        >>> import tensorflow as tf
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
        >>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = [
        ...     "There is only one actor who is 45 years old",
        ...     "There are 3 actors which played in more than 60 movies",
        ... ]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> labels = tf.convert_to_tensor([1, 0])  # 1 means entailed, 0 means refuted

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        outputs = self.tapas(
            input_ids=input_ids,
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
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "tapas", None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


""" TAPAS utilities."""


class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


# Beginning of everything related to segmented tensors


class IndexMap:
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index.

        Args:
          indices: <int32> Tensor of indices, same shape as `values`.
          num_segments: <int32> Scalar tensor, the number of segments. All elements
            in a batched segmented tensor must have the same number of segments (although many segments can be empty).
          batch_dims: Python integer, the number of batch dimensions. The first
            `batch_dims` dimensions of a SegmentedTensor are treated as batch dimensions. Segments in different batch
            elements are always distinct even if they have the same index.
        """
        self.indices = tf.convert_to_tensor(indices)
        self.num_segments = tf.convert_to_tensor(num_segments)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return tf.shape(self.indices)[: self.batch_dims]


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segements` * `inner_index.num_segments`.

        Args:
          outer_index: IndexMap.
          inner_index: IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(
                inner_index.indices
                + outer_index.indices * tf.cast(inner_index.num_segments, inner_index.indices.dtype)
            ),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=tf.math.floordiv(index.indices, self.inner_index.num_segments),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=tf.math.floormod(index.indices, self.inner_index.num_segments),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def gather(values, index, name="segmented_gather"):
    """
    Gathers from `values` using the index map. For each element in the domain of the index map this operation looks up
    a value for that index in `values`. Two elements from the same segment always get assigned the same value.

    Args:
      values: [B1, ..., Bn, num_segments, V1, ...] Tensor with segment values.
      index: [B1, ..., Bn, I1, ..., Ik] IndexMap.
      name: Name for the TensorFlow operation.

    Returns:
      [B1, ..., Bn, I1, ..., Ik, V1, ...] Tensor with the gathered values.
    """
    return tf.gather(values, index.indices, batch_dims=index.batch_dims, name=name)


def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map to a 1d index map. This operation relabels the segments to keep batch elements
    distinct. The k-th batch element will have indices shifted by `num_segments` * (k - 1). The result is a tensor with
    `num_segments` multiplied by the number of elements in the batch.

    Args:
      index: IndexMap to flatten.
      name: Name for the TensorFlow operation.

    Returns:
      The flattened IndexMap.
    """
    batch_size = tf.reduce_prod(index.batch_shape())
    offset = tf.range(batch_size) * index.num_segments
    offset = tf.reshape(offset, index.batch_shape())
    for _ in range(index.batch_dims, index.indices.shape.rank):
        offset = tf.expand_dims(offset, -1)

    indices = tf.cast(offset, index.indices.dtype) + index.indices
    return IndexMap(indices=tf.reshape(indices, [-1]), num_segments=index.num_segments * batch_size, batch_dims=0)


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (`tf.Tensor`):
            Batch shape
        num_segments (`int`):
            Number of segments
        name (`str`, *optional*, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = tf.convert_to_tensor(batch_shape)
    batch_shape.shape.assert_has_rank(1)
    num_segments = tf.convert_to_tensor(num_segments)
    num_segments.shape.assert_has_rank(0)

    indices = tf.range(num_segments)
    shape = tf.concat([tf.ones_like(batch_shape, dtype=tf.int32), tf.expand_dims(num_segments, axis=0)], axis=0)
    indices = tf.reshape(indices, shape)
    multiples = tf.concat([batch_shape, [1]], axis=0)
    indices = tf.tile(indices, multiples)
    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=batch_shape.shape.as_list()[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.

    Args:
        values (`tf.Tensor`):
            Tensor with segment values.
        index (`IndexMap`):
            IndexMap.
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (`str`):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = tf.shape(values)[index.indices.shape.rank :]
    flattened_shape = tf.concat([[-1], vector_shape], axis=0)
    flat_values = tf.reshape(values, flattened_shape)
    segment_means = segment_reduce_fn(
        data=flat_values, segment_ids=flat_index.indices, num_segments=flat_index.num_segments
    )

    # Unflatten the values.
    new_shape = tf.concat([index.batch_shape(), [index.num_segments], vector_shape], axis=0)
    output_values = tf.reshape(segment_means, new_shape)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments. Outputs 0 for empty segments. This operations computes the mean over segments,
    with support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be a mean of vectors
        rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, tf.math.unsorted_segment_mean, name)


def reduce_sum(values, index, name="segmented_reduce_sum"):
    """
    Sums a tensor over its segments. Outputs 0 for empty segments. This operations computes the sum over segments, with
    support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be a sum of vectors
        rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, tf.math.unsorted_segment_sum, name)


def reduce_max(values, index, name="segmented_reduce_max"):
    """
    Computes the maximum over segments. This operations computes the maximum over segments, with support for:

      - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
      - Vectorization using the last dimension [V1, V2, ...]. If they are present the output will be an element-wise
        maximum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
      values: [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..] tensor of values to be
        averaged.
      index: IndexMap [B1, B2, ..., Bn, I1, .., Ik] index defining the segments.
      name: Name for the TensorFlow ops.

    Returns:
      A pair (output_values, output_index) where `output_values` is a tensor of shape [B1, B2, ..., Bn, num_segments,
      V1, V2, ..] and `index` is an IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, tf.math.unsorted_segment_max, name)


def reduce_min(values, index, name="segmented_reduce_min"):
    """Computes the minimum over segments."""
    return _segment_reduce(values, index, tf.math.unsorted_segment_min, name)


def _single_column_cell_selection_loss(token_logits, column_logits, labels, cell_index, col_index, cell_mask):
    """
    Computes the loss for cell selection constrained to a single column. The loss is a hierarchical log-likelihood. The
    model first predicts a column and then selects cells within that column (conditioned on the column). Cells outside
    the selected column are never selected.

    Args:
        token_logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the logits per token.
        column_logits (`tf.Tensor` of shape `(batch_size, max_num_cols)`):
            Tensor containing the logits per column.
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Labels per token.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        col_index (`IndexMap`):
            Index that groups tokens into columns.
        cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).

    Returns:
        selection_loss_per_example (`tf.Tensor` of shape `(batch_size,)`): Loss for each example. logits (`tf.Tensor`
        of shape `(batch_size, sequence_length)`): New logits which are only allowed to select cells in a single
        column. Logits outside of the most likely column according to *column_logits* will be set to a very low value
        (such that the probabilities are 0).
    """
    # First find the column we should select. We use the column with maximum
    # number of selected cells.
    labels_per_column, _ = reduce_sum(tf.cast(labels, tf.float32), col_index)
    column_label = tf.argmax(labels_per_column, axis=-1, output_type=tf.int32)
    # Check if there are no selected cells in the column. In that case the model
    # should predict the special column id 0, which means "select nothing".
    no_cell_selected = tf.equal(tf.reduce_max(labels_per_column, axis=-1), 0)
    column_label = tf.where(no_cell_selected, tf.zeros_like(column_label), column_label)

    column_dist = tfp.distributions.Categorical(logits=column_logits)
    column_loss_per_example = -column_dist.log_prob(column_label)

    # Reduce the labels and logits to per-cell from per-token.
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    labels_per_cell, labels_index = reduce_max(tf.cast(labels, tf.int32), cell_index)

    # Mask for the selected column.
    column_id_for_cells = cell_index.project_inner(labels_index).indices
    column_mask = tf.cast(tf.equal(column_id_for_cells, tf.expand_dims(column_label, axis=1)), tf.float32)

    # Compute the log-likelihood for cells, but only for the selected column.
    cell_dist = tfp.distributions.Bernoulli(logits=logits_per_cell)
    cell_log_prob = cell_dist.log_prob(labels_per_cell)
    cell_loss = -tf.reduce_sum(cell_log_prob * column_mask * cell_mask, axis=1)
    # We need to normalize the loss by the number of cells in the column.
    cell_loss /= tf.reduce_sum(column_mask * cell_mask, axis=1) + EPSILON_ZERO_DIVISION

    selection_loss_per_example = column_loss_per_example
    selection_loss_per_example += tf.where(no_cell_selected, tf.zeros_like(selection_loss_per_example), cell_loss)

    # Set the probs outside the selected column (selected by the *model*)
    # to 0. This ensures backwards compatibility with models that select
    # cells from multiple columns.
    selected_column_id = tf.argmax(column_logits, axis=-1, output_type=tf.int32)
    selected_column_mask = tf.cast(
        tf.equal(column_id_for_cells, tf.expand_dims(selected_column_id, axis=-1)), tf.float32
    )
    # Never select cells with the special column id 0.
    selected_column_mask = tf.where(
        tf.equal(column_id_for_cells, 0), tf.zeros_like(selected_column_mask), selected_column_mask
    )
    logits_per_cell += CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    logits = gather(logits_per_cell, cell_index)

    return selection_loss_per_example, logits


def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    """
    Finds examples where the model should select cells with no aggregation.

    Returns a mask that determines for which examples should the model select answers directly from the table, without
    any aggregation function. If the answer is a piece of text the case is unambiguous as aggregation functions only
    apply to numbers. If the answer is a number but does not appear in the table then we must use some aggregation
    case. The ambiguous case is when the answer is a number that also appears in the table. In this case we use the
    aggregation function probabilities predicted by the model to decide whether to select or aggregate. The threshold
    for this is a hyperparameter *cell_selection_preference*

    Args:
        answer (`tf.Tensor` of shape `(batch_size, )`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        pooled_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Output of the pooler (BertPooler) on top of the encoder layer.
        cell_selection_preference (`float`):
            Preference for cell selection in ambiguous cases.
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Labels per token. aggregation_classifier (`torch.nn.Linear`): Aggregation head

    Returns:
        aggregate_mask (`tf.Tensor` of shape `(batch_size,)`): A mask set to 1 for examples that should use aggregation
        functions.
    """
    # tf.Tensor(batch_size,)
    aggregate_mask_init = tf.cast(tf.logical_not(tf.math.is_nan(answer)), tf.float32)
    logits_aggregation = aggregation_classifier(pooled_output)
    dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
    # Index 0 corresponds to "no aggregation".
    aggregation_ops_total_mass = tf.reduce_sum(dist_aggregation.probs_parameter()[:, 1:], axis=1)
    # Cell selection examples according to current model.
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference
    # Examples with non-empty cell selection supervision.
    is_cell_supervision_available = tf.reduce_sum(labels, axis=1) > 0
    aggregate_mask = tf.where(
        tf.logical_and(is_pred_cell_selection, is_cell_supervision_available),
        tf.zeros_like(aggregate_mask_init, dtype=tf.float32),
        aggregate_mask_init,
    )
    aggregate_mask = tf.stop_gradient(aggregate_mask)
    return aggregate_mask


def _calculate_aggregation_loss_known(
    logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
):
    """
    Calculates aggregation loss when its type is known during training.

    In the weakly supervised setting, the only known information is that for cell selection examples, "no aggregation"
    should be predicted. For other examples (those that require aggregation), no loss is accumulated. In the setting
    where aggregation type is always known, standard cross entropy loss is accumulated for all examples

    Args:
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`tf.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`tf.Tensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.

    Returns:
        aggregation_loss_known (`tf.Tensor` of shape `(batch_size,)`): Aggregation loss (when its type is known during
        training) per example.
    """
    if use_answer_as_supervision:
        # Prepare "no aggregation" targets for cell selection examples.
        target_aggregation = tf.zeros_like(aggregate_mask, dtype=tf.int32)
    else:
        # Use aggregation supervision as the target.
        target_aggregation = aggregation_labels

    one_hot_labels = tf.one_hot(target_aggregation, depth=num_aggregation_labels, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits_aggregation, axis=-1)

    # <float32>[batch_size]
    per_example_aggregation_intermediate = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    if use_answer_as_supervision:
        # Accumulate loss only for examples requiring cell selection
        # (no aggregation).
        return per_example_aggregation_intermediate * (1 - aggregate_mask)
    else:
        return per_example_aggregation_intermediate


def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
    """
    Calculates aggregation loss in the case of answer supervision.

    Args:
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`tf.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions

    Returns:
        aggregation_loss_unknown (`tf.Tensor` of shape `(batch_size,)`): Aggregation loss (in case of answer
        supervision) per example.
    """
    dist_aggregation = tfp.distributions.Categorical(logits=logits_aggregation)
    # Index 0 corresponds to "no aggregation".
    aggregation_ops_total_mass = tf.reduce_sum(dist_aggregation.probs_parameter()[:, 1:], axis=1)
    # Predict some aggregation in case of an answer that needs aggregation.
    # This increases the probability of all aggregation functions, in a way
    # similar to MML, but without considering whether the function gives the
    # correct answer.
    return -tf.math.log(aggregation_ops_total_mass) * aggregate_mask


def _calculate_aggregation_loss(
    logits_aggregation,
    aggregate_mask,
    aggregation_labels,
    use_answer_as_supervision,
    num_aggregation_labels,
    aggregation_loss_weight,
):
    """
    Calculates the aggregation loss per example.

    Args:
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`tf.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`tf.Tensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            Importance weight for the aggregation loss.

    Returns:
        aggregation_loss (`tf.Tensor` of shape `(batch_size,)`): Aggregation loss per example.
    """
    per_example_aggregation_loss = _calculate_aggregation_loss_known(
        logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
    )

    if use_answer_as_supervision:
        # Add aggregation loss for numeric answers that need aggregation.
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return aggregation_loss_weight * per_example_aggregation_loss


def _calculate_expected_result(
    dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    """
    Calculates the expected result given cell and aggregation probabilities.

    Args:
        dist_per_cell (`tfp.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the hyperparameters of the model

    Returns:
        expected_result (`tf.Tensor` of shape `(batch_size,)`): The expected result per example.
    """
    if config.use_gumbel_for_cells:
        gumbel_dist = tfp.distributions.RelaxedBernoulli(
            # The token logits where already divided by the temperature and used for
            # computing cell selection errors so we need to multiply it again here
            config.temperature,
            logits=dist_per_cell.logits_parameter() * config.temperature,
        )
        scaled_probability_per_cell = gumbel_dist.sample()
    else:
        scaled_probability_per_cell = dist_per_cell.probs_parameter()

    # <float32>[batch_size, seq_length]
    scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float
    count_result = tf.reduce_sum(scaled_probability_per_cell, axis=1)
    numeric_values_masked = tf.where(
        tf.math.is_nan(numeric_values), tf.zeros_like(numeric_values), numeric_values
    )  # Mask non-numeric table values to zero.
    sum_result = tf.reduce_sum(scaled_probability_per_cell * numeric_values_masked, axis=1)
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        # The sum of all probabilities exept that correspond to other cells
        ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) - scaled_probability_per_cell + 1
        average_result = tf.reduce_sum(numeric_values_masked * scaled_probability_per_cell / ex, axis=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        # The sum of all probabilities exept that correspond to other cells
        ex = tf.reduce_sum(scaled_probability_per_cell, axis=1, keepdims=True) - scaled_probability_per_cell + 1
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        var = tf.reduce_sum(pointwise_var, axis=1, keepdims=True) - pointwise_var
        multiplier = (var / tf.math.square(ex) + 1) / ex
        average_result = tf.reduce_sum(numeric_values_masked * scaled_probability_per_cell * multiplier, axis=1)
    else:
        raise ValueError("Invalid average_approximation_function: %s", config.average_approximation_function)

    if config.use_gumbel_for_aggregation:
        gumbel_dist = tfp.distributions.RelaxedOneHotCategorical(
            config.aggregation_temperature, logits=logits_aggregation[:, 1:]
        )
        # <float32>[batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = gumbel_dist.sample()
    else:
        # <float32>[batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = stable_softmax(logits_aggregation[:, 1:] / config.aggregation_temperature, axis=-1)
    all_results = tf.concat(
        [
            tf.expand_dims(sum_result, axis=1),
            tf.expand_dims(average_result, axis=1),
            tf.expand_dims(count_result, axis=1),
        ],
        axis=1,
    )
    expected_result = tf.reduce_sum(all_results * aggregation_op_only_probs, axis=1)
    return expected_result


def _calculate_regression_loss(
    answer,
    aggregate_mask,
    dist_per_cell,
    numeric_values,
    numeric_values_scale,
    input_mask_float,
    logits_aggregation,
    config,
):
    """
    Calculates the regression loss per example.

    Args:
        answer (`tf.Tensor` of shape `(batch_size,)`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        aggregate_mask (`tf.Tensor` of shape `(batch_size,)`):
            A mask set to 1 for examples that should use aggregation functions.
        dist_per_cell (`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`tf.Tensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`tf.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the parameters of the model

    Returns:
        per_example_answer_loss_scaled (`tf.Tensor` of shape `(batch_size,)`): Scales answer loss for each example in
        the batch. large_answer_loss_mask (`tf.Tensor` of shape `(batch_size,)`): A mask which is 1 for examples for
        which their answer loss is larger than the answer_loss_cutoff.
    """
    # float32 (batch_size,)
    expected_result = _calculate_expected_result(
        dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # <float32>[batch_size]
    answer_masked = tf.where(tf.math.is_nan(answer), tf.zeros_like(answer), answer)

    if config.use_normalized_answer_loss:
        normalizer = tf.stop_gradient(
            tf.math.maximum(tf.math.abs(expected_result), tf.math.abs(answer_masked)) + EPSILON_ZERO_DIVISION
        )
        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        per_example_answer_loss = tf.compat.v1.losses.huber_loss(
            normalized_answer_masked * aggregate_mask,
            normalized_expected_result * aggregate_mask,
            delta=tf.cast(1.0, tf.float32),
            reduction=tf.losses.Reduction.NONE,
        )
    else:
        per_example_answer_loss = tf.compat.v1.losses.huber_loss(
            answer_masked * aggregate_mask,
            expected_result * aggregate_mask,
            delta=tf.cast(config.huber_loss_delta, tf.float32),
            reduction=tf.losses.Reduction.NONE,
        )
    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = tf.ones_like(per_example_answer_loss, dtype=tf.float32)
    else:
        large_answer_loss_mask = tf.where(
            per_example_answer_loss > config.answer_loss_cutoff,
            tf.zeros_like(per_example_answer_loss, dtype=tf.float32),
            tf.ones_like(per_example_answer_loss, dtype=tf.float32),
        )
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)
    return per_example_answer_loss_scaled, large_answer_loss_mask


__all__ = [
    "TFTapasForMaskedLM",
    "TFTapasForQuestionAnswering",
    "TFTapasForSequenceClassification",
    "TFTapasModel",
    "TFTapasPreTrainedModel",
]
