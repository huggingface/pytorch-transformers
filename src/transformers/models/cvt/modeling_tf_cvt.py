# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""TF 2.0 Cvt model."""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
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
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_cvt import CvtConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CvtConfig"


@dataclass
class TFBaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`tf.Tensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
    """

    last_hidden_state: tf.Tensor = None
    cls_token_value: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None


class TFCvtDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x: tf.Tensor, training=None):
        if self.drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=self.compute_dtype)
        random_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * random_tensor


class TFCvtEmbeddings(keras.layers.Layer):
    """Construct the Convolutional Token Embeddings."""

    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        dropout_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.convolution_embeddings = TFCvtConvEmbeddings(
            config,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=embed_dim,
            stride=stride,
            padding=padding,
            name="convolution_embeddings",
        )
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution_embeddings", None) is not None:
            with tf.name_scope(self.convolution_embeddings.name):
                self.convolution_embeddings.build(None)


class TFCvtConvEmbeddings(keras.layers.Layer):
    """Image to Convolution Embeddings. This convolutional operation aims to model local spatial contexts."""

    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=padding)
        self.patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.projection = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="valid",
            data_format="channels_last",
            kernel_initializer=get_initializer(config.initializer_range),
            name="projection",
        )
        # Using the same default epsilon as PyTorch
        self.normalization = keras.layers.LayerNormalization(epsilon=1e-5, name="normalization")
        self.num_channels = num_channels
        self.embed_dim = embed_dim

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        pixel_values = self.projection(self.padding(pixel_values))

        # "batch_size, height, width, num_channels -> batch_size, (height*width), num_channels"
        batch_size, height, width, num_channels = shape_list(pixel_values)
        hidden_size = height * width
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, hidden_size, num_channels))
        pixel_values = self.normalization(pixel_values)

        # "batch_size, (height*width), num_channels -> batch_size, height, width, num_channels"
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, height, width, num_channels))
        return pixel_values

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, self.embed_dim])


class TFCvtSelfAttentionConvProjection(keras.layers.Layer):
    """Convolutional projection layer."""

    def __init__(self, config: CvtConfig, embed_dim: int, kernel_size: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=padding)
        self.convolution = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="valid",
            strides=stride,
            use_bias=False,
            name="convolution",
            groups=embed_dim,
        )
        # Using the same default epsilon as PyTorch, TF uses (1 - pytorch momentum)
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution(self.padding(hidden_state))
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.embed_dim])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.embed_dim])


class TFCvtSelfAttentionLinearProjection(keras.layers.Layer):
    """Linear projection layer used to flatten tokens into 1D."""

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # "batch_size, height, width, num_channels -> batch_size, (height*width), num_channels"
        batch_size, height, width, num_channels = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))
        return hidden_state


class TFCvtSelfAttentionProjection(keras.layers.Layer):
    """Convolutional Projection for Attention."""

    def __init__(
        self,
        config: CvtConfig,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        projection_method: str = "dw_bn",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if projection_method == "dw_bn":
            self.convolution_projection = TFCvtSelfAttentionConvProjection(
                config, embed_dim, kernel_size, stride, padding, name="convolution_projection"
            )
        self.linear_projection = TFCvtSelfAttentionLinearProjection()

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution_projection(hidden_state, training=training)
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution_projection", None) is not None:
            with tf.name_scope(self.convolution_projection.name):
                self.convolution_projection.build(None)


class TFCvtSelfAttention(keras.layers.Layer):
    """
    Self-attention layer. A depth-wise separable convolution operation (Convolutional Projection), is applied for
    query, key, and value embeddings.
    """

    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        with_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = embed_dim**-0.5
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.convolution_projection_query = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_q,
            padding_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
            name="convolution_projection_query",
        )
        self.convolution_projection_key = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_key",
        )
        self.convolution_projection_value = TFCvtSelfAttentionProjection(
            config,
            embed_dim,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_value",
        )

        self.projection_query = keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_query",
        )
        self.projection_key = keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_key",
        )
        self.projection_value = keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=qkv_bias,
            bias_initializer="zeros",
            name="projection_value",
        )
        self.dropout = keras.layers.Dropout(attention_drop_rate)

    def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor:
        batch_size, hidden_size, _ = shape_list(hidden_state)
        head_dim = self.embed_dim // self.num_heads
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, self.num_heads, head_dim))
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        return hidden_state

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        if self.with_cls_token:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # "batch_size, (height*width), num_channels -> batch_size, height, width, num_channels"
        batch_size, hidden_size, num_channels = shape_list(hidden_state)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))

        key = self.convolution_projection_key(hidden_state, training=training)
        query = self.convolution_projection_query(hidden_state, training=training)
        value = self.convolution_projection_value(hidden_state, training=training)

        if self.with_cls_token:
            query = tf.concat((cls_token, query), axis=1)
            key = tf.concat((cls_token, key), axis=1)
            value = tf.concat((cls_token, value), axis=1)

        head_dim = self.embed_dim // self.num_heads

        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        attention_score = tf.matmul(query, key, transpose_b=True) * self.scale
        attention_probs = stable_softmax(logits=attention_score, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context = tf.matmul(attention_probs, value)
        # "batch_size, num_heads, hidden_size, head_dim -> batch_size, hidden_size, (num_heads*head_dim)"
        _, _, hidden_size, _ = shape_list(context)
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, hidden_size, self.num_heads * head_dim))
        return context

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution_projection_query", None) is not None:
            with tf.name_scope(self.convolution_projection_query.name):
                self.convolution_projection_query.build(None)
        if getattr(self, "convolution_projection_key", None) is not None:
            with tf.name_scope(self.convolution_projection_key.name):
                self.convolution_projection_key.build(None)
        if getattr(self, "convolution_projection_value", None) is not None:
            with tf.name_scope(self.convolution_projection_value.name):
                self.convolution_projection_value.build(None)
        if getattr(self, "projection_query", None) is not None:
            with tf.name_scope(self.projection_query.name):
                self.projection_query.build([None, None, self.embed_dim])
        if getattr(self, "projection_key", None) is not None:
            with tf.name_scope(self.projection_key.name):
                self.projection_key.build([None, None, self.embed_dim])
        if getattr(self, "projection_value", None) is not None:
            with tf.name_scope(self.projection_value.name):
                self.projection_value.build([None, None, self.embed_dim])


class TFCvtSelfOutput(keras.layers.Layer):
    """Output of the Attention layer ."""

    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = keras.layers.Dropout(drop_rate)
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.embed_dim])


class TFCvtAttention(keras.layers.Layer):
    """Attention layer. First chunk of the convolutional transformer block."""

    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        with_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = TFCvtSelfAttention(
            config,
            num_heads,
            embed_dim,
            kernel_size,
            stride_q,
            stride_kv,
            padding_q,
            padding_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
            name="attention",
        )
        self.dense_output = TFCvtSelfOutput(config, embed_dim, drop_rate, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False):
        self_output = self.attention(hidden_state, height, width, training=training)
        attention_output = self.dense_output(self_output, training=training)
        return attention_output

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


class TFCvtIntermediate(keras.layers.Layer):
    """Intermediate dense layer. Second chunk of the convolutional transformer block."""

    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=int(embed_dim * mlp_ratio),
            kernel_initializer=get_initializer(config.initializer_range),
            activation="gelu",
            name="dense",
        )
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        hidden_state = self.dense(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.embed_dim])


class TFCvtOutput(keras.layers.Layer):
    """
    Output of the Convolutional Transformer Block (last chunk). It consists of a MLP and a residual connection.
    """

    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, drop_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = keras.layers.Dropout(drop_rate)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

    def call(self, hidden_state: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        hidden_state = hidden_state + input_tensor
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, int(self.embed_dim * self.mlp_ratio)])


class TFCvtLayer(keras.layers.Layer):
    """
    Convolutional Transformer Block composed by attention layers, normalization and multi-layer perceptrons (mlps). It
    consists of 3 chunks : an attention layer, an intermediate dense layer and an output layer. This corresponds to the
    `Block` class in the original implementation.
    """

    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        mlp_ratio: float,
        drop_path_rate: float,
        with_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = TFCvtAttention(
            config,
            num_heads,
            embed_dim,
            kernel_size,
            stride_q,
            stride_kv,
            padding_q,
            padding_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
            name="attention",
        )
        self.intermediate = TFCvtIntermediate(config, embed_dim, mlp_ratio, name="intermediate")
        self.dense_output = TFCvtOutput(config, embed_dim, mlp_ratio, drop_rate, name="output")
        # Using `layers.Activation` instead of `tf.identity` to better control `training` behaviour.
        self.drop_path = (
            TFCvtDropPath(drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )
        # Using the same default epsilon as PyTorch
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_before")
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_after")
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        # in Cvt, layernorm is applied before self-attention
        attention_output = self.attention(self.layernorm_before(hidden_state), height, width, training=training)
        attention_output = self.drop_path(attention_output, training=training)

        # first residual connection
        hidden_state = attention_output + hidden_state

        # in Cvt, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.dense_output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output, training=training)
        return layer_output

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
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.embed_dim])
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.embed_dim])


class TFCvtStage(keras.layers.Layer):
    """
    Cvt stage (encoder block). Each stage has 2 parts :
    - (1) A Convolutional Token Embedding layer
    - (2) A Convolutional Transformer Block (layer).
    The classification token is added only in the last stage.

    Args:
        config ([`CvtConfig`]): Model configuration class.
        stage (`int`): Stage number.
    """

    def __init__(self, config: CvtConfig, stage: int, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stage = stage
        if self.config.cls_token[self.stage]:
            self.cls_token = self.add_weight(
                shape=(1, 1, self.config.embed_dim[-1]),
                initializer=get_initializer(self.config.initializer_range),
                trainable=True,
                name="cvt.encoder.stages.2.cls_token",
            )

        self.embedding = TFCvtEmbeddings(
            self.config,
            patch_size=config.patch_sizes[self.stage],
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],
            stride=config.patch_stride[self.stage],
            embed_dim=config.embed_dim[self.stage],
            padding=config.patch_padding[self.stage],
            dropout_rate=config.drop_rate[self.stage],
            name="embedding",
        )

        drop_path_rates = tf.linspace(0.0, config.drop_path_rate[self.stage], config.depth[stage])
        drop_path_rates = [x.numpy().item() for x in drop_path_rates]
        self.layers = [
            TFCvtLayer(
                config,
                num_heads=config.num_heads[self.stage],
                embed_dim=config.embed_dim[self.stage],
                kernel_size=config.kernel_qkv[self.stage],
                stride_q=config.stride_q[self.stage],
                stride_kv=config.stride_kv[self.stage],
                padding_q=config.padding_q[self.stage],
                padding_kv=config.padding_kv[self.stage],
                qkv_projection_method=config.qkv_projection_method[self.stage],
                qkv_bias=config.qkv_bias[self.stage],
                attention_drop_rate=config.attention_drop_rate[self.stage],
                drop_rate=config.drop_rate[self.stage],
                mlp_ratio=config.mlp_ratio[self.stage],
                drop_path_rate=drop_path_rates[self.stage],
                with_cls_token=config.cls_token[self.stage],
                name=f"layers.{j}",
            )
            for j in range(config.depth[self.stage])
        ]

    def call(self, hidden_state: tf.Tensor, training: bool = False):
        cls_token = None
        hidden_state = self.embedding(hidden_state, training)

        # "batch_size, height, width, num_channels -> batch_size, (height*width), num_channels"
        batch_size, height, width, num_channels = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))

        if self.config.cls_token[self.stage]:
            cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
            hidden_state = tf.concat((cls_token, hidden_state), axis=1)

        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width, training=training)
            hidden_state = layer_outputs

        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # "batch_size, (height*width), num_channels -> batch_size, height, width, num_channels"
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))
        return hidden_state, cls_token

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedding", None) is not None:
            with tf.name_scope(self.embedding.name):
                self.embedding.build(None)
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFCvtEncoder(keras.layers.Layer):
    """
    Convolutional Vision Transformer encoder. CVT has 3 stages of encoder blocks with their respective number of layers
    (depth) being 1, 2 and 10.

    Args:
        config ([`CvtConfig`]): Model configuration class.
    """

    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stages = [
            TFCvtStage(config, stage_idx, name=f"stages.{stage_idx}") for stage_idx in range(len(config.depth))
        ]

    def call(
        self,
        pixel_values: TFModelInputType,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values
        # When running on CPU, `keras.layers.Conv2D` doesn't support (batch_size, num_channels, height, width)
        # as input format. So change the input format to (batch_size, height, width, num_channels).
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 3, 1))

        cls_token = None
        for _, (stage_module) in enumerate(self.stages):
            hidden_state, cls_token = stage_module(hidden_state, training=training)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        # Change back to (batch_size, num_channels, height, width) format to have uniformity in the modules
        hidden_state = tf.transpose(hidden_state, perm=(0, 3, 1, 2))
        if output_hidden_states:
            all_hidden_states = tuple([tf.transpose(hs, perm=(0, 3, 1, 2)) for hs in all_hidden_states])

        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "stages", None) is not None:
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFCvtMainLayer(keras.layers.Layer):
    """Construct the Cvt model."""

    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFCvtEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)


class TFCvtPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CvtConfig
    base_model_prefix = "cvt"
    main_input_name = "pixel_values"


TFCVT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    </Tip>

    Args:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

TFCVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`CvtImageProcessor.__call__`]
            for details.

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
    "The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.",
    TFCVT_START_DOCSTRING,
)
class TFCvtModel(TFCvtPreTrainedModel):
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.cvt = TFCvtMainLayer(config, name="cvt")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithCLSToken, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtModel.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.cvt(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return (outputs[0],) + outputs[1:]

        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=outputs.last_hidden_state,
            cls_token_value=outputs.cls_token_value,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "cvt", None) is not None:
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)


@add_start_docstrings(
    """
    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    TFCVT_START_DOCSTRING,
)
class TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.cvt = TFCvtMainLayer(config, name="cvt")
        # Using same default epsilon as in the original implementation.
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm")

        # Classifier head
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="classifier",
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtForImageClassification.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""

        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            # rearrange "batch_size, num_channels, height, width -> batch_size, (height*width), num_channels"
            batch_size, num_channels, height, width = shape_list(sequence_output)
            sequence_output = tf.reshape(sequence_output, shape=(batch_size, num_channels, height * width))
            sequence_output = tf.transpose(sequence_output, perm=(0, 2, 1))
            sequence_output = self.layernorm(sequence_output)

        sequence_output_mean = tf.reduce_mean(sequence_output, axis=1)
        logits = self.classifier(sequence_output_mean)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "cvt", None) is not None:
            with tf.name_scope(self.cvt.name):
                self.cvt.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.embed_dim[-1]])
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.embed_dim[-1]])


__all__ = ["TFCvtForImageClassification", "TFCvtModel", "TFCvtPreTrainedModel"]
