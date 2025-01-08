# coding=utf-8
# Copyright 2022 Meta Platforms and The HuggingFace Inc. team. All rights reserved.
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
"""TF 2.0 Data2Vec Vision model."""

from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFSemanticSegmenterOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_data2vec_vision import Data2VecVisionConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Data2VecVisionConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-vision-base"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/data2vec-vision-base-ft1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "remote control, remote"


@dataclass
class TFData2VecVisionModelOutputWithPooling(TFBaseModelOutputWithPooling):
    """
    Class for outputs of [`TFData2VecVisionModel`].

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


class TFData2VecVisionDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class TFData2VecVisionEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.patch_embeddings = TFData2VecVisionPatchEmbeddings(config, name="patch_embeddings")
        self.num_patches = self.patch_embeddings.num_patches
        self.config = config

        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        if self.config.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="mask_token",
            )
        else:
            self.mask_token = None

        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.add_weight(
                shape=(1, self.num_patches + 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="position_embeddings",
            )
        else:
            self.position_embeddings = None

        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None = None) -> tf.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, projection_dim = shape_list(embeddings)

        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))

        if bool_masked_pos is not None:
            mask_tokens = tf.broadcast_to(self.mask_token, (batch_size, seq_len, projection_dim))
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos[..., None]
            w = tf.cast(w, mask_tokens.dtype)
            # since TF doesn't support eager tensor assignment
            embeddings = embeddings * (1 - w) + mask_tokens * w

        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class TFData2VecVisionPatchEmbeddings(keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels

        self.projection = keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # following torch.nn.Linear
            bias_initializer="zeros",
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])

        return tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])


class TFData2VecVisionSelfAttention(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
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
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
            use_bias=False,
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        if window_size:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
        else:
            self.relative_position_bias = None
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.sqrt_att_head_size

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            # Passing `0.0` to the `relative_position_bias()` layer because otherwise Keras
            # might complain about `Layer.call()` not being invoked properly. In this case this input
            # i.e., 0.0 is not going to be used in any calculations so we're safe.
            attention_scores = attention_scores + self.relative_position_bias(0.0)[None, ...]

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

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
        if getattr(self, "relative_position_bias", None) is not None:
            with tf.name_scope(self.relative_position_bias.name):
                self.relative_position_bias.build(None)


class TFData2VecVisionSelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFData2VecVisionLayer instead of here (as is the case with other models), due
    to the layernorm applied before each block.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, gamma=None, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFData2VecVisionAttention(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFData2VecVisionSelfAttention(config, window_size=window_size, name="attention")
        self.dense_output = TFData2VecVisionSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->Data2VecVision
class TFData2VecVisionIntermediate(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
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


class TFData2VecVisionOutput(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


class TFData2VecVisionLayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config

        self.attention = TFData2VecVisionAttention(config, window_size=window_size, name="attention")
        self.intermediate = TFData2VecVisionIntermediate(config, name="intermediate")
        self.data2vec_output = TFData2VecVisionOutput(config, name="output")

        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        # Using `layers.Activation` instead of `tf.identity` to better control `training`
        # behaviour.
        self.drop_path = (
            TFData2VecVisionDropPath(drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        self.init_values = config.layer_scale_init_value

    def build(self, input_shape: tf.TensorShape = None):
        if self.init_values > 0:
            self.lambda_1 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_1",
            )
            self.lambda_2 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_2",
            )
            self.lambda_1.assign(self.init_values * tf.ones((self.config.hidden_size)))
            self.lambda_2.assign(self.init_values * tf.ones((self.config.hidden_size)))
        else:
            self.lambda_1, self.lambda_2 = None, None

        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "data2vec_output", None) is not None:
            with tf.name_scope(self.data2vec_output.name):
                self.data2vec_output.build(None)
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_attention_outputs = self.attention(
            # in Data2VecVision, layernorm is applied before self-attention
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            training=training,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Data2VecVision, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.data2vec_output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Taken and modified from here:
# https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/beit.py#L28
class TFData2VecVisionRelativePositionBias(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.window_size = window_size
        # +3 for cls_token_pos_len
        # window_size can be something like (14, 14)
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        self.relative_position_index = self.get_position_index()

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(self.num_relative_distance, self.config.num_attention_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )  # [2*Wh-1 * 2*Ww-1, nH]
        # cls to token & token 2 cls & cls to cls

        super().build(input_shape)

    def get_position_index(self):
        # get pair-wise relative position index for each token inside the window
        xx, yy = tf.meshgrid(range(self.window_size[0]), range(self.window_size[1]))
        coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
        coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])  # [Wh*Ww, Wh*Ww, 2]

        xx = (relative_coords[:, :, 0] + self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        yy = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords = tf.stack([xx, yy], axis=-1)

        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]

        top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 3
        )
        left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 2
        )
        corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (self.num_relative_distance - 1)

        left_corner = tf.concat([corner, left], axis=0)
        relative_position_index = tf.concat([top, relative_position_index], axis=0)
        relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [Wh*Ww + 1, Wh*Ww + 1]
        return relative_position_index

    def call(self, inputs=None) -> tf.Tensor:
        relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=0)
        return tf.transpose(relative_position_bias, [2, 0, 1])


class TFData2VecVisionEncoder(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
        else:
            self.relative_position_bias = None

        # stochastic depth decay rule
        dpr = list(tf.linspace(0.0, config.drop_path_rate, config.num_hidden_layers))
        self.layer = [
            TFData2VecVisionLayer(
                config,
                window_size=window_size if config.use_relative_position_bias else None,
                drop_path_rate=dpr[i],
                name=f"layer_._{i}",
            )
            for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # Passing `0.0` to the `relative_position_bias()` layer because otherwise Keras
            # might complain about `Layer.call()` not being invoked properly. In this case this input
            # i.e., 0.0 is not going to be used in any calculations so we're safe.
            relative_position_bias = (
                self.relative_position_bias(0.0) if self.relative_position_bias is not None else None
            )
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "relative_position_bias", None) is not None:
            with tf.name_scope(self.relative_position_bias.name):
                self.relative_position_bias.build(None)
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFData2VecVisionMainLayer(keras.layers.Layer):
    config_class = Data2VecVisionConfig

    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.add_pooling_layer = add_pooling_layer

        self.embeddings = TFData2VecVisionEmbeddings(config, name="embeddings")
        self.encoder = TFData2VecVisionEncoder(
            config, window_size=self.embeddings.patch_embeddings.patch_shape, name="encoder"
        )
        self.layernorm = (
            tf.identity
            if config.use_mean_pooling
            else keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        )

        # We are setting the `data_format` like so because from here on we will revert to the
        # NCHW output format
        self.pooler = TFData2VecVisionPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(pixel_values, bool_masked_pos, training=training)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return TFData2VecVisionModelOutputWithPooling(
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
        if getattr(self, "layernorm", None) is not None:
            if hasattr(self.layernorm, "name"):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFData2VecVisionPooler(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = (
            keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
            if config.use_mean_pooling
            else None
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(tf.reduce_mean(patch_tokens, axis=1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layernorm", None) is not None:
            if hasattr(self.layernorm, "name"):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))


class TFData2VecVisionPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecVisionConfig
    base_model_prefix = "data2vec_vision"
    main_input_name = "pixel_values"
    _keys_to_ignore_on_load_unexpected = [r"relative_position_index"]


DATA2VEC_VISION_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.).

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

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`Data2VecVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

DATA2VEC_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.

        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Data2VecVision Model transformer outputting raw hidden-states without any specific head on top.",
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionModel(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        self.data2vec_vision = TFData2VecVisionMainLayer(
            config, add_pooling_layer=add_pooling_layer, name="data2vec_vision"
        )

    def get_input_embeddings(self):
        return self.data2vec_vision.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFData2VecVisionModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
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
        if getattr(self, "data2vec_vision", None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)


@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForImageClassification(TFData2VecVisionPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=True, name="data2vec_vision")

        # Classifier head
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, tuple]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
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
        if getattr(self, "data2vec_vision", None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


class TFData2VecVisionConvModule(keras.layers.Layer):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: str = "valid",
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=bias,
            dilation_rate=dilation,
            name="conv",
        )
        self.bn = keras.layers.BatchNormalization(name="bn", momentum=0.9, epsilon=1e-5)
        self.activation = tf.nn.relu
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv", None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, None, self.in_channels])
        if getattr(self, "bn", None) is not None:
            with tf.name_scope(self.bn.name):
                self.bn.build((None, None, None, self.out_channels))


class TFAdaptiveAvgPool2D(keras.layers.Layer):
    def __init__(self, output_dims: Tuple[int, int], input_ordering: str = "NHWC", **kwargs):
        super().__init__(**kwargs)
        self.output_dims = output_dims
        self.input_ordering = input_ordering
        if input_ordering not in ("NCHW", "NHWC"):
            raise ValueError("Unrecognized input_ordering, should be 'NCHW' or 'NHWC'!")
        self.h_axis = input_ordering.index("H")
        self.w_axis = input_ordering.index("W")

    def pseudo_1d_pool(self, inputs: tf.Tensor, h_pooling: bool):
        # Figure out which axis we're pooling on
        if h_pooling:
            axis = self.h_axis
            output_dim = self.output_dims[0]
        else:
            axis = self.w_axis
            output_dim = self.output_dims[1]
        input_dim = inputs.shape[axis]

        # Figure out the potential pooling windows
        # This is the key idea - the torch op always uses only two
        # consecutive pooling window sizes, like 3 and 4. Therefore,
        # if we pool with both possible sizes, we simply need to gather
        # the 'correct' pool at each position to reimplement the torch op.
        small_window = math.ceil(input_dim / output_dim)
        big_window = small_window + 1
        if h_pooling:
            output_dim = self.output_dims[0]
            small_window_shape = (small_window, 1)
            big_window_shape = (big_window, 1)
        else:
            output_dim = self.output_dims[1]
            small_window_shape = (1, small_window)
            big_window_shape = (1, big_window)

        # For resizes to 1, or integer resizes, we can take quick shortcuts
        if output_dim == input_dim:
            return inputs
        elif output_dim == 1:
            return tf.reduce_mean(inputs, axis=axis, keepdims=True)
        elif input_dim % output_dim == 0:
            return tf.nn.avg_pool2d(
                inputs,
                ksize=small_window_shape,
                strides=small_window_shape,
                padding="VALID",
                data_format=self.input_ordering,
            )
        # When upscaling by an integer factor we can also take a quick shortcut
        elif output_dim > input_dim and output_dim % input_dim == 0:
            return tf.repeat(inputs, repeats=output_dim // input_dim, axis=axis)

        # For non-integer resizes, we pool with both possible window sizes and concatenate them
        if output_dim < input_dim:
            small_pool = tf.nn.avg_pool2d(
                inputs, ksize=small_window_shape, strides=1, padding="VALID", data_format=self.input_ordering
            )
            big_pool = tf.nn.avg_pool2d(
                inputs, ksize=big_window_shape, strides=1, padding="VALID", data_format=self.input_ordering
            )
            both_pool = tf.concat([small_pool, big_pool], axis=axis)
        else:
            # When we're actually upscaling instead, then we build the pools a bit differently
            small_pool = inputs
            big_pool = tf.nn.avg_pool2d(
                inputs, ksize=big_window_shape, strides=1, padding="VALID", data_format=self.input_ordering
            )
            both_pool = tf.concat([small_pool, big_pool], axis=axis)

        # We compute vectors of the start and end positions for each pooling window
        # Each (start, end) pair here corresponds to a single output position
        window_starts = tf.math.floor((tf.range(output_dim, dtype=tf.float32) * input_dim) / output_dim)
        window_starts = tf.cast(window_starts, tf.int64)
        window_ends = tf.math.ceil((tf.range(1, output_dim + 1, dtype=tf.float32) * input_dim) / output_dim)
        window_ends = tf.cast(window_ends, tf.int64)

        # pool_selector is a boolean array of shape (output_dim,) where 1 indicates that output position
        # has a big receptive field and 0 indicates that that output position has a small receptive field
        pool_selector = tf.cast(window_ends - window_starts - small_window, tf.bool)

        # Since we concatenated the small and big pools, we need to do a bit of
        # pointer arithmetic to get the indices of the big pools
        small_indices = window_starts
        big_indices = window_starts + small_pool.shape[axis]

        # Finally, we use the pool_selector to generate a list of indices, one per output position
        gather_indices = tf.where(pool_selector, big_indices, small_indices)

        # Gathering from those indices yields the final, correct pooling
        return tf.gather(both_pool, gather_indices, axis=axis)

    def call(self, inputs: tf.Tensor):
        if self.input_ordering == "NHWC":
            input_shape = inputs.shape[1:3]
        else:
            input_shape = inputs.shape[2:]

        # We break the task down into each possible case
        # Firstly, if we're resizing down to 1, it's just tf.reduce_mean
        if self.output_dims[0] == self.output_dims[1] == 1:
            if self.input_ordering == "NHWC":
                reduce_dims = [1, 2]
            else:
                reduce_dims = [2, 3]
            return tf.reduce_mean(inputs, axis=reduce_dims, keepdims=True)
        # Secondly, if we're resizing by an integer factor on both dimensions, we can take a quick shortcut
        elif input_shape[0] % self.output_dims[0] == 0 and input_shape[1] % self.output_dims[1] == 0:
            h_resize = int(input_shape[0] // self.output_dims[0])
            w_resize = int(input_shape[1] // self.output_dims[1])
            return tf.nn.avg_pool2d(
                inputs,
                ksize=(h_resize, w_resize),
                strides=(h_resize, w_resize),
                padding="VALID",
                data_format=self.input_ordering,
            )
        else:
            # Finally, if we can't take the shortcut, we do a 1D pool on each axis. pseudo_1d_pool will take a shortcut
            # for dimensions where an integer resize is possible. It can also handle upscaling.
            h_pooled = self.pseudo_1d_pool(inputs, h_pooling=True)
            return self.pseudo_1d_pool(h_pooled, h_pooling=False)


class TFData2VecVisionPyramidPoolingModule(keras.layers.Layer):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        channels (int): Channels after modules, before conv_seg.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer_list = []
        for idx, pool_scale in enumerate(pool_scales):
            pool_scale = pool_scale if isinstance(pool_scale, collections.abc.Iterable) else (pool_scale, pool_scale)
            self.layer_list.append(
                [
                    TFAdaptiveAvgPool2D(output_dims=pool_scale),
                    TFData2VecVisionConvModule(
                        in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, name=f"{idx}.1"
                    ),
                ]
            )

    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        ppm_outs = []
        inputs = x

        for ppm in self.layer_list:
            for layer_module in ppm:
                ppm_out = layer_module(x)
                x = ppm_out

            upsampled_ppm_out = tf.image.resize(ppm_out, size=shape_list(inputs)[1:-1], method="bilinear")
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

    def build(self, input_shape=None):
        for layer in self.layer_list:
            for layer_module in layer:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


class TFData2VecVisionUperHead(keras.layers.Layer):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.classifier = keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")

        # PSP Module
        self.psp_modules = TFData2VecVisionPyramidPoolingModule(
            self.pool_scales, self.in_channels[-1], self.channels, name="psp_modules"
        )
        self.bottleneck = TFData2VecVisionConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding="same",
            name="bottleneck",
        )
        # FPN Module
        self.lateral_convs = []
        self.fpn_convs = []
        for idx, in_channels in enumerate(self.in_channels[:-1]):  # skip the top layer
            l_conv = TFData2VecVisionConvModule(
                in_channels, out_channels=self.channels, kernel_size=1, name=f"lateral_convs.{idx}"
            )
            fpn_conv = TFData2VecVisionConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding="same",
                name=f"fpn_convs.{idx}",
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = TFData2VecVisionConvModule(
            in_channels=len(self.in_channels) * self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding="same",
            name="fpn_bottleneck",
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = tf.concat(psp_outs, axis=-1)
        output = self.bottleneck(psp_outs)

        return output

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = shape_list(laterals[i - 1])[1:-1]
            laterals[i - 1] = laterals[i - 1] + tf.image.resize(laterals[i], size=prev_shape, method="bilinear")

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = tf.image.resize(fpn_outs[i], size=shape_list(fpn_outs[0])[1:-1], method="bilinear")
        fpn_outs = tf.concat(fpn_outs, axis=-1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.channels])
        if getattr(self, "psp_modules", None) is not None:
            with tf.name_scope(self.psp_modules.name):
                self.psp_modules.build(None)
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        if getattr(self, "fpn_bottleneck", None) is not None:
            with tf.name_scope(self.fpn_bottleneck.name):
                self.fpn_bottleneck.build(None)
        for layer in self.lateral_convs:
            with tf.name_scope(layer.name):
                layer.build(None)
        for layer in self.fpn_convs:
            with tf.name_scope(layer.name):
                layer.build(None)


class TFData2VecVisionFCNHead(keras.layers.Layer):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented from
    [FCNNet](https://arxiv.org/abs/1411.4038).

    Args:
        config (Data2VecVisionConfig): Configuration.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        convs = []
        convs.append(
            TFData2VecVisionConvModule(
                in_channels=self.in_channels,
                out_channels=self.channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                name="convs.0",
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                TFData2VecVisionConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=kernel_size,
                    padding="same",
                    dilation=dilation,
                    name=f"conv_module_{i+2}",
                )
            )
        if self.num_convs == 0:
            self.convs = [tf.identity]
        else:
            self.convs = convs
        if self.concat_input:
            self.conv_cat = TFData2VecVisionConvModule(
                self.in_channels + self.channels,
                out_channels=self.channels,
                kernel_size=kernel_size,
                padding="same",
                name="conv_cat",
            )

        self.classifier = keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = hidden_states
        for layer_module in self.convs:
            output = layer_module(output)
        if self.concat_input:
            output = self.conv_cat(tf.concat([hidden_states, output], axis=-1))
        output = self.classifier(output)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.channels])
        if getattr(self, "conv_cat", None) is not None:
            with tf.name_scope(self.conv_cat.name):
                self.conv_cat.build(None)


@add_start_docstrings(
    """
    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForSemanticSegmentation(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=False, name="data2vec_vision")

        # FPNs
        self.fpn1 = [
            keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.0"),
            keras.layers.BatchNormalization(name="fpn1.1", momentum=0.9, epsilon=1e-5),
            keras.layers.Activation("gelu"),
            keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.3"),
        ]
        self.fpn2 = [keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn2.0")]

        self.fpn3 = tf.identity
        self.fpn4 = keras.layers.MaxPool2D(pool_size=2, strides=2)

        # Semantic segmentation head(s)
        self.decode_head = TFData2VecVisionUperHead(config, name="decode_head")
        self.auxiliary_head = (
            TFData2VecVisionFCNHead(config, name="auxiliary_head") if config.use_auxiliary_head else None
        )

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        if len(shape_list(labels)) > 3:
            label_interp_shape = shape_list(labels)[1:-1]
        else:
            label_interp_shape = shape_list(labels)[-2:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = tf.image.resize(auxiliary_logits, size=label_interp_shape, method="bilinear")
        # compute weighted loss
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        # Copied from https://www.tensorflow.org/text/tutorials/transformer#loss_and_metrics.
        # Utility to mask the index to ignore during computing the loss.
        def masked_loss(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, self.config.semantic_loss_ignore_index))
            loss_ = loss_fct(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            reduced_masked_loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        main_loss = masked_loss(labels, upsampled_logits)
        auxiliary_loss = masked_loss(labels, upsampled_auxiliary_logits)
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TFSemanticSegmenterOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFData2VecVisionForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        >>> model = TFData2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.data2vec_vision(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        patch_resolution = self.config.image_size // self.config.patch_size

        def reshape_features(x):
            # We do it this way so TF can always infer the non-batch dims at compile time
            x = tf.reshape(x, (-1, patch_resolution, patch_resolution, self.config.hidden_size))
            return x

        features = [reshape_features(x[:, 1:, :]) for x in features]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for module in ops[0]:
            features[0] = module(features[0])
        features[1] = ops[1][0](features[1])
        for i in range(len(features[2:])):
            features[i + 2] = ops[i + 2](features[i + 2])

        logits = self.decode_head(features)
        # Tranpose the logits to maintain consistency in the output formats.
        transposed_logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSemanticSegmenterOutput(
            loss=loss,
            logits=transposed_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "data2vec_vision", None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
        if getattr(self, "decode_head", None) is not None:
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)
        if getattr(self, "auxiliary_head", None) is not None:
            with tf.name_scope(self.auxiliary_head.name):
                self.auxiliary_head.build(None)
        if getattr(self, "fpn1", None) is not None:
            with tf.name_scope(self.fpn1[0].name):
                self.fpn1[0].build([None, None, None, self.config.hidden_size])
            with tf.name_scope(self.fpn1[1].name):
                self.fpn1[1].build((None, None, None, self.config.hidden_size))
            with tf.name_scope(self.fpn1[3].name):
                self.fpn1[3].build([None, None, None, self.config.hidden_size])
        if getattr(self, "fpn2", None) is not None:
            with tf.name_scope(self.fpn2[0].name):
                self.fpn2[0].build([None, None, None, self.config.hidden_size])


__all__ = [
    "TFData2VecVisionForImageClassification",
    "TFData2VecVisionForSemanticSegmentation",
    "TFData2VecVisionModel",
    "TFData2VecVisionPreTrainedModel",
]
