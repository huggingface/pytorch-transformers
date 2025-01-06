# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team.
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
"""TF 2.0 LayoutLMv3 model."""

from __future__ import annotations

import collections
import math
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config


_CONFIG_FOR_DOC = "LayoutLMv3Config"

_DUMMY_INPUT_IDS = [
    [7, 6, 1],
    [1, 2, 0],
]

_DUMMY_BBOX = [
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
]


LARGE_NEGATIVE = -1e8


class TFLayoutLMv3PatchEmbeddings(keras.layers.Layer):
    """LayoutLMv3 image (patch) embeddings."""

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        patch_sizes = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.proj = keras.layers.Conv2D(
            filters=config.hidden_size,
            kernel_size=patch_sizes,
            strides=patch_sizes,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(config.initializer_range),
            name="proj",
        )
        self.hidden_size = config.hidden_size
        self.num_patches = (config.input_size**2) // (patch_sizes[0] * patch_sizes[1])
        self.config = config

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])

        embeddings = self.proj(pixel_values)
        embeddings = tf.reshape(embeddings, (-1, self.num_patches, self.hidden_size))
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, None, self.config.num_channels])


class TFLayoutLMv3TextEmbeddings(keras.layers.Layer):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        self.token_type_embeddings = keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="token_type_embeddings",
        )
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.padding_token_index = config.pad_token_id
        self.position_embeddings = keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )
        self.x_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="x_position_embeddings",
        )
        self.y_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="y_position_embeddings",
        )
        self.h_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="h_position_embeddings",
        )
        self.w_position_embeddings = keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="w_position_embeddings",
        )
        self.max_2d_positions = config.max_2d_position_embeddings
        self.config = config

    def calculate_spatial_position_embeddings(self, bbox: tf.Tensor) -> tf.Tensor:
        try:
            left_position_ids = bbox[:, :, 0]
            upper_position_ids = bbox[:, :, 1]
            right_position_ids = bbox[:, :, 2]
            lower_position_ids = bbox[:, :, 3]
        except IndexError as exception:
            raise IndexError("Bounding box is not of shape (batch_size, seq_length, 4).") from exception

        try:
            left_position_embeddings = self.x_position_embeddings(left_position_ids)
            upper_position_embeddings = self.y_position_embeddings(upper_position_ids)
            right_position_embeddings = self.x_position_embeddings(right_position_ids)
            lower_position_embeddings = self.y_position_embeddings(lower_position_ids)
        except IndexError as exception:
            raise IndexError(
                f"The `bbox` coordinate values should be within 0-{self.max_2d_positions} range."
            ) from exception

        max_position_id = self.max_2d_positions - 1
        h_position_embeddings = self.h_position_embeddings(
            tf.clip_by_value(bbox[:, :, 3] - bbox[:, :, 1], 0, max_position_id)
        )
        w_position_embeddings = self.w_position_embeddings(
            tf.clip_by_value(bbox[:, :, 2] - bbox[:, :, 0], 0, max_position_id)
        )

        # LayoutLMv1 sums the spatial embeddings, but LayoutLMv3 concatenates them.
        spatial_position_embeddings = tf.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embds: tf.Tensor) -> tf.Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded, so just generate sequential position
        ids.
        """
        input_shape = tf.shape(inputs_embds)
        sequence_length = input_shape[1]
        start_index = self.padding_token_index + 1
        end_index = self.padding_token_index + sequence_length + 1
        position_ids = tf.range(start_index, end_index, dtype=tf.int32)
        batch_size = input_shape[0]
        position_ids = tf.reshape(position_ids, (1, sequence_length))
        position_ids = tf.tile(position_ids, (batch_size, 1))
        return position_ids

    def create_position_ids_from_input_ids(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_token_index + 1.
        """
        mask = tf.cast(tf.not_equal(input_ids, self.padding_token_index), input_ids.dtype)
        position_ids = tf.cumsum(mask, axis=1) * mask
        position_ids = position_ids + self.padding_token_index
        return position_ids

    def create_position_ids(self, input_ids: tf.Tensor, inputs_embeds: tf.Tensor) -> tf.Tensor:
        if input_ids is None:
            return self.create_position_ids_from_inputs_embeds(inputs_embeds)
        else:
            return self.create_position_ids_from_input_ids(input_ids)

    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids, inputs_embeds)

        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=position_ids.dtype)

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.word_embeddings.input_dim)
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        embeddings += spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "word_embeddings", None) is not None:
            with tf.name_scope(self.word_embeddings.name):
                self.word_embeddings.build(None)
        if getattr(self, "token_type_embeddings", None) is not None:
            with tf.name_scope(self.token_type_embeddings.name):
                self.token_type_embeddings.build(None)
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, "position_embeddings", None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                self.position_embeddings.build(None)
        if getattr(self, "x_position_embeddings", None) is not None:
            with tf.name_scope(self.x_position_embeddings.name):
                self.x_position_embeddings.build(None)
        if getattr(self, "y_position_embeddings", None) is not None:
            with tf.name_scope(self.y_position_embeddings.name):
                self.y_position_embeddings.build(None)
        if getattr(self, "h_position_embeddings", None) is not None:
            with tf.name_scope(self.h_position_embeddings.name):
                self.h_position_embeddings.build(None)
        if getattr(self, "w_position_embeddings", None) is not None:
            with tf.name_scope(self.w_position_embeddings.name):
                self.w_position_embeddings.build(None)


class TFLayoutLMv3SelfAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_score_normaliser = math.sqrt(self.attention_head_size)

        self.query = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.config = config

    def transpose_for_scores(self, x: tf.Tensor):
        shape = tf.shape(x)
        new_shape = (
            shape[0],  # batch_size
            shape[1],  # seq_length
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # batch_size, num_heads, seq_length, attention_head_size

    def cogview_attention(self, attention_scores: tf.Tensor, alpha: Union[float, int] = 32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original keras.layers.Softmax(axis=-1)(attention_scores). Seems the new
        attention_probs will result in a slower speed and a little bias. Can use
        tf.debugging.assert_near(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison. The
        smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = tf.expand_dims(tf.reduce_max(scaled_attention_scores, axis=-1), axis=-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return tf.math.softmax(new_attention_scores, axis=-1)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = None,
        rel_2d_pos: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        normalised_query_layer = query_layer / self.attention_score_normaliser
        transposed_key_layer = tf.transpose(
            key_layer, perm=[0, 1, 3, 2]
        )  # batch_size, num_heads, attention_head_size, seq_length
        attention_scores = tf.matmul(normalised_query_layer, transposed_key_layer)

        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / self.attention_score_normaliser
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / self.attention_score_normaliser

        if attention_mask is not None:
            # Apply the attention mask (is precomputed for all layers in TFLayoutLMv3Model call() function)
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        # Use the trick of CogView paper to stabilize training.
        attention_probs = self.cogview_attention(attention_scores)

        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to.
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(
            context_layer, perm=[0, 2, 1, 3]
        )  # batch_size, seq_length, num_heads, attention_head_size
        shape = tf.shape(context_layer)
        context_layer = tf.reshape(
            context_layer, (shape[0], shape[1], self.all_head_size)
        )  # batch_size, seq_length, num_heads * attention_head_size

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

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


# Copied from models.roberta.modeling_tf_roberta.TFRobertaSelfOutput
class TFLayoutLMv3SelfOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
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


class TFLayoutLMv3Attention(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFLayoutLMv3SelfAttention(config, name="self")
        self.self_output = TFLayoutLMv3SelfOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = None,
        rel_2d_pos: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        self_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos,
            training=training,
        )
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)


# Copied from models.roberta.modeling_tf_bert.TFRobertaIntermediate
class TFLayoutLMv3Intermediate(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
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


# Copied from models.roberta.modeling_tf_bert.TFRobertaOutput
class TFLayoutLMv3Output(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
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


class TFLayoutLMv3Layer(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLayoutLMv3Attention(config, name="attention")
        self.intermediate = TFLayoutLMv3Intermediate(config, name="intermediate")
        self.bert_output = TFLayoutLMv3Output(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None,
        head_mask: tf.Tensor | None,
        output_attentions: bool,
        rel_pos: tf.Tensor | None = None,
        rel_2d_pos: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            training=training,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs
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


class TFLayoutLMv3Encoder(keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFLayoutLMv3Layer(config, name=f"layer.{i}") for i in range(config.num_hidden_layers)]

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_bias",
            )

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_x_bias",
            )
            self.rel_pos_y_bias = keras.layers.Dense(
                units=config.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
                name="rel_pos_y_bias",
            )

    def relative_position_bucket(self, relative_positions: tf.Tensor, num_buckets: int, max_distance: int):
        # the negative relative positions are assigned to the interval [0, num_buckets / 2]
        # we deal with this by assigning absolute relative positions to the interval [0, num_buckets / 2]
        # and then offsetting the positive relative positions by num_buckets / 2 at the end
        num_buckets = num_buckets // 2
        buckets = tf.abs(relative_positions)

        # half of the buckets are for exact increments in positions
        max_exact_buckets = num_buckets // 2
        is_small = buckets < max_exact_buckets

        # the other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        buckets_log_ratio = tf.math.log(tf.cast(buckets, tf.float32) / max_exact_buckets)
        distance_log_ratio = math.log(max_distance / max_exact_buckets)
        buckets_big_offset = (
            buckets_log_ratio / distance_log_ratio * (num_buckets - max_exact_buckets)
        )  # scale is [0, num_buckets - max_exact_buckets]
        buckets_big = max_exact_buckets + buckets_big_offset  # scale is [max_exact_buckets, num_buckets]
        buckets_big = tf.cast(buckets_big, buckets.dtype)
        buckets_big = tf.minimum(buckets_big, num_buckets - 1)

        return (tf.cast(relative_positions > 0, buckets.dtype) * num_buckets) + tf.where(
            is_small, buckets, buckets_big
        )

    def _cal_pos_emb(
        self,
        dense_layer: keras.layers.Dense,
        position_ids: tf.Tensor,
        num_buckets: int,
        max_distance: int,
    ):
        rel_pos_matrix = tf.expand_dims(position_ids, axis=-2) - tf.expand_dims(position_ids, axis=-1)
        rel_pos = self.relative_position_bucket(rel_pos_matrix, num_buckets, max_distance)
        rel_pos_one_hot = tf.one_hot(rel_pos, depth=num_buckets, dtype=self.compute_dtype)
        embedding = dense_layer(rel_pos_one_hot)
        # batch_size, seq_length, seq_length, num_heads --> batch_size, num_heads, seq_length, seq_length
        embedding = tf.transpose(embedding, [0, 3, 1, 2])
        embedding = tf.cast(embedding, dtype=self.compute_dtype)
        return embedding

    def _cal_1d_pos_emb(self, position_ids: tf.Tensor):
        return self._cal_pos_emb(self.rel_pos_bias, position_ids, self.rel_pos_bins, self.max_rel_pos)

    def _cal_2d_pos_emb(self, bbox: tf.Tensor):
        position_coord_x = bbox[:, :, 0]  # left
        position_coord_y = bbox[:, :, 3]  # bottom
        rel_pos_x = self._cal_pos_emb(
            self.rel_pos_x_bias,
            position_coord_x,
            self.rel_2d_pos_bins,
            self.max_rel_2d_pos,
        )
        rel_pos_y = self._cal_pos_emb(
            self.rel_pos_y_bias,
            position_coord_y,
            self.rel_2d_pos_bins,
            self.max_rel_2d_pos,
        )
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def call(
        self,
        hidden_states: tf.Tensor,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        position_ids: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[
        TFBaseModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
                training=training,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_dict:
            return TFBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            return tuple(
                value for value in [hidden_states, all_hidden_states, all_self_attentions] if value is not None
            )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "rel_pos_bias", None) is not None:
            with tf.name_scope(self.rel_pos_bias.name):
                self.rel_pos_bias.build([None, None, self.rel_pos_bins])
        if getattr(self, "rel_pos_x_bias", None) is not None:
            with tf.name_scope(self.rel_pos_x_bias.name):
                self.rel_pos_x_bias.build([None, None, self.rel_2d_pos_bins])
        if getattr(self, "rel_pos_y_bias", None) is not None:
            with tf.name_scope(self.rel_pos_y_bias.name):
                self.rel_pos_y_bias.build([None, None, self.rel_2d_pos_bins])
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFLayoutLMv3MainLayer(keras.layers.Layer):
    config_class = LayoutLMv3Config

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        if config.text_embed:
            self.embeddings = TFLayoutLMv3TextEmbeddings(config, name="embeddings")

        if config.visual_embed:
            self.patch_embed = TFLayoutLMv3PatchEmbeddings(config, name="patch_embed")
            self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

            if config.has_relative_attention_bias or config.has_spatial_attention_bias:
                image_size = config.input_size // config.patch_size
                self.init_visual_bbox(image_size=(image_size, image_size))

            self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        self.encoder = TFLayoutLMv3Encoder(config, name="encoder")

    def build(self, input_shape=None):
        if self.config.visual_embed:
            image_size = self.config.input_size // self.config.patch_size
            self.cls_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
                name="cls_token",
            )
            self.pos_embed = self.add_weight(
                shape=(1, image_size * image_size + 1, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
                name="pos_embed",
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "patch_embed", None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.hidden_size])

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.word_embeddings.weight = value

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def init_visual_bbox(self, image_size: Tuple[int, int], max_len: int = 1000):
        # We should not hardcode max_len to 1000, but it is done by the reference implementation,
        # so we keep it for compatibility with the pretrained weights. The more correct approach
        # would have been to pass on max_len=config.max_2d_position_embeddings - 1.
        height, width = image_size

        visual_bbox_x = tf.range(0, max_len * (width + 1), max_len) // width
        visual_bbox_x = tf.expand_dims(visual_bbox_x, axis=0)
        visual_bbox_x = tf.tile(visual_bbox_x, [width, 1])  # (width, width + 1)

        visual_bbox_y = tf.range(0, max_len * (height + 1), max_len) // height
        visual_bbox_y = tf.expand_dims(visual_bbox_y, axis=1)
        visual_bbox_y = tf.tile(visual_bbox_y, [1, height])  # (height + 1, height)

        visual_bbox = tf.stack(
            [visual_bbox_x[:, :-1], visual_bbox_y[:-1], visual_bbox_x[:, 1:], visual_bbox_y[1:]],
            axis=-1,
        )
        visual_bbox = tf.reshape(visual_bbox, [-1, 4])

        cls_token_box = tf.constant([[1, 1, max_len - 1, max_len - 1]], dtype=tf.int32)
        self.visual_bbox = tf.concat([cls_token_box, visual_bbox], axis=0)

    def calculate_visual_bbox(self, batch_size: int, dtype: tf.DType):
        visual_bbox = tf.expand_dims(self.visual_bbox, axis=0)
        visual_bbox = tf.tile(visual_bbox, [batch_size, 1, 1])
        visual_bbox = tf.cast(visual_bbox, dtype=dtype)
        return visual_bbox

    def embed_image(self, pixel_values: tf.Tensor) -> tf.Tensor:
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size = tf.shape(embeddings)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        # add position embeddings
        if getattr(self, "pos_embed", None) is not None:
            embeddings += self.pos_embed

        embeddings = self.norm(embeddings)
        return embeddings

    def get_extended_attention_mask(self, attention_mask: tf.Tensor) -> tf.Tensor:
        # Adapted from transformers.modelling_utils.ModuleUtilsMixin.get_extended_attention_mask

        n_dims = len(attention_mask.shape)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if n_dims == 3:
            extended_attention_mask = tf.expand_dims(attention_mask, axis=1)
        elif n_dims == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length].
            # Make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length].
            extended_attention_mask = tf.expand_dims(attention_mask, axis=1)  # (batch_size, 1, seq_length)
            extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)  # (batch_size, 1, 1, seq_length)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape}).")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, self.compute_dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * LARGE_NEGATIVE

        return extended_attention_mask

    def get_head_mask(self, head_mask: tf.Tensor | None) -> Union[tf.Tensor, List[tf.Tensor | None]]:
        if head_mask is None:
            return [None] * self.config.num_hidden_layers

        n_dims = tf.rank(head_mask)
        if n_dims == 1:
            # Gets a tensor with masks for each head (H).
            head_mask = tf.expand_dims(head_mask, axis=0)  # 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=0)  # 1, 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=-1)  # 1, 1, num_heads, 1
            head_mask = tf.expand_dims(head_mask, axis=-1)  # 1, 1, num_heads, 1, 1
            head_mask = tf.tile(
                head_mask, [self.config.num_hidden_layers, 1, 1, 1, 1]
            )  # seq_length, 1, num_heads, 1, 1
        elif n_dims == 2:
            # Gets a tensor with masks for each layer (L) and head (H).
            head_mask = tf.expand_dims(head_mask, axis=1)  # seq_length, 1, num_heads
            head_mask = tf.expand_dims(head_mask, axis=-1)  # seq_length, 1, num_heads, 1
            head_mask = tf.expand_dims(head_mask, axis=-1)  # seq_length, 1, num_heads, 1, 1
        elif n_dims != 5:
            raise ValueError(f"Wrong shape for head_mask (shape {head_mask.shape}).")
        assert tf.rank(head_mask) == 5, f"Got head_mask rank of {tf.rank(head_mask)}, but require 5."
        head_mask = tf.cast(head_mask, self.compute_dtype)
        return head_mask

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[
        TFBaseModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # This method can be called with a variety of modalities:
        # 1. text + layout
        # 2. text + layout + image
        # 3. image
        # The complexity of this method is mostly just due to handling of these different modalities.

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None:
            input_shape = tf.shape(input_ids)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
        elif pixel_values is not None:
            batch_size = tf.shape(pixel_values)[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        # Determine which integer dtype to use.
        if input_ids is not None:
            int_dtype = input_ids.dtype
        elif bbox is not None:
            int_dtype = bbox.dtype
        elif attention_mask is not None:
            int_dtype = attention_mask.dtype
        elif token_type_ids is not None:
            int_dtype = token_type_ids.dtype
        else:
            int_dtype = tf.int32

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = tf.ones((batch_size, seq_length), dtype=int_dtype)
            if token_type_ids is None:
                token_type_ids = tf.zeros((batch_size, seq_length), dtype=int_dtype)
            if bbox is None:
                bbox = tf.zeros((batch_size, seq_length, 4), dtype=int_dtype)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                training=training,
            )

        final_bbox = None
        final_position_ids = None
        if pixel_values is not None:
            # embed image
            visual_embeddings = self.embed_image(pixel_values)

            # calculate attention mask
            visual_attention_mask = tf.ones((batch_size, tf.shape(visual_embeddings)[1]), dtype=int_dtype)
            if attention_mask is None:
                attention_mask = visual_attention_mask
            else:
                attention_mask = tf.concat([attention_mask, visual_attention_mask], axis=1)

            # calculate bounding boxes
            if self.config.has_spatial_attention_bias:
                visual_bbox = self.calculate_visual_bbox(batch_size, int_dtype)
                if bbox is None:
                    final_bbox = visual_bbox
                else:
                    final_bbox = tf.concat([bbox, visual_bbox], axis=1)

            # calculate position IDs
            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                visual_position_ids = tf.range(0, tf.shape(visual_embeddings)[1], dtype=int_dtype)
                visual_position_ids = tf.expand_dims(visual_position_ids, axis=0)
                visual_position_ids = tf.tile(visual_position_ids, [batch_size, 1])

                if input_ids is not None or inputs_embeds is not None:
                    position_ids = tf.expand_dims(tf.range(0, seq_length, dtype=int_dtype), axis=0)
                    position_ids = tf.tile(position_ids, [batch_size, 1])
                    final_position_ids = tf.concat([position_ids, visual_position_ids], axis=1)
                else:
                    final_position_ids = visual_position_ids

            # calculate embeddings
            if input_ids is None and inputs_embeds is None:
                embedding_output = visual_embeddings
            else:
                embedding_output = tf.concat([embedding_output, visual_embeddings], axis=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output, training=training)

        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_relative_attention_bias:
                position_ids = tf.expand_dims(tf.range(0, seq_length, dtype=int_dtype), axis=0)
                position_ids = tf.tile(position_ids, [batch_size, 1])
                final_position_ids = position_ids

            if self.config.has_spatial_attention_bias:
                final_bbox = bbox

        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x seq_length x seq_length
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFLayoutLMv3PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv3Config
    base_model_prefix = "layoutlmv3"

    @property
    def input_signature(self):
        sig = super().input_signature
        sig["bbox"] = tf.TensorSpec((None, None, 4), tf.int32, name="bbox")
        return sig


LAYOUTLMV3_START_DOCSTRING = r"""
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
        config ([`LayoutLMv3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLMV3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        bbox (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Batch of document images. Each image is divided into patches of shape `(num_channels, config.patch_size,
            config.patch_size)` and the total number of patches (=`patch_sequence_length`) equals to `((height /
            config.patch_size) * (width / config.patch_size))`.

        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
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
    "The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3Model(TFLayoutLMv3PreTrainedModel):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[
        TFBaseModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModel
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModel.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="tf")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
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
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)


class TFLayoutLMv3ClassificationHead(keras.layers.Layer):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            config.hidden_size,
            activation="tanh",
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(
            classifier_dropout,
            name="dropout",
        )
        self.out_proj = keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj",
        )
        self.config = config

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        outputs = self.dropout(inputs, training=training)
        outputs = self.dense(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = self.out_proj(outputs)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForSequenceClassification(TFLayoutLMv3PreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        self.classifier = TFLayoutLMv3ClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[
        TFSequenceClassifierOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForSequenceClassification
        >>> from datasets import load_dataset
        >>> import tensorflow as tf

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="tf")
        >>> sequence_label = tf.convert_to_tensor([1])

        >>> outputs = model(**encoding, labels=sequence_label)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
            training=training,
        )
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output, training=training)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
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
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)


@add_start_docstrings(
    """
    LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
    for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
    [SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
    [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForTokenClassification(TFLayoutLMv3PreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels

        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        if config.num_labels < 10:
            self.classifier = keras.layers.Dense(
                config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="classifier",
            )
        else:
            self.classifier = TFLayoutLMv3ClassificationHead(config, name="classifier")
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        bbox: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[
        TFTokenClassifierOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="tf")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            training=training,
        )
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
    compute `span start logits` and `span end logits`).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class TFLayoutLMv3ForQuestionAnswering(TFLayoutLMv3PreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)

        self.num_labels = config.num_labels

        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name="layoutlmv3")
        self.qa_outputs = TFLayoutLMv3ClassificationHead(config, name="qa_outputs")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        start_positions: tf.Tensor | None = None,
        end_positions: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        bbox: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[
        TFQuestionAnsweringModelOutput,
        Tuple[tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForQuestionAnswering
        >>> from datasets import load_dataset
        >>> import tensorflow as tf

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> question = "what's his name?"
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, question, words, boxes=boxes, return_tensors="tf")
        >>> start_positions = tf.convert_to_tensor([1])
        >>> end_positions = tf.convert_to_tensor([3])

        >>> outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
            training=training,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output, training=training)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)

        loss = None

        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, logits=(start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layoutlmv3", None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build(None)


__all__ = [
    "TFLayoutLMv3ForQuestionAnswering",
    "TFLayoutLMv3ForSequenceClassification",
    "TFLayoutLMv3ForTokenClassification",
    "TFLayoutLMv3Model",
    "TFLayoutLMv3PreTrainedModel",
]
