# coding=utf-8
# Copyright 2022 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
#
# Original license: https://github.com/apple/ml-cvnets/blob/main/LICENSE
"""TensorFlow 2.0 MobileViT model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutputWithNoAttention,
    TFSemanticSegmenterOutputWithNoAttention,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MobileViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class TFMobileViTConvLayer(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        logger.warning(
            f"\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish "
            "to train/fine-tune this model, you need a GPU or a TPU"
        )

        padding = int((kernel_size - 1) / 2) * dilation
        self.padding = keras.layers.ZeroPadding2D(padding)

        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        self.convolution = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            dilation_rate=dilation,
            groups=groups,
            use_bias=bias,
            name="convolution",
        )

        if use_normalization:
            self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="normalization")
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                self.activation = get_tf_activation(use_activation)
            elif isinstance(config.hidden_act, str):
                self.activation = get_tf_activation(config.hidden_act)
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        padded_features = self.padding(features)
        features = self.convolution(padded_features)
        if self.normalization is not None:
            features = self.normalization(features, training=training)
        if self.activation is not None:
            features = self.activation(features)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, "normalization", None) is not None:
            if hasattr(self.normalization, "name"):
                with tf.name_scope(self.normalization.name):
                    self.normalization.build([None, None, None, self.out_channels])


class TFMobileViTInvertedResidual(keras.layers.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = TFMobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, name="expand_1x1"
        )

        self.conv_3x3 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
            name="conv_3x3",
        )

        self.reduce_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            name="reduce_1x1",
        )

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = features

        features = self.expand_1x1(features, training=training)
        features = self.conv_3x3(features, training=training)
        features = self.reduce_1x1(features, training=training)

        return residual + features if self.use_residual else features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "expand_1x1", None) is not None:
            with tf.name_scope(self.expand_1x1.name):
                self.expand_1x1.build(None)
        if getattr(self, "conv_3x3", None) is not None:
            with tf.name_scope(self.conv_3x3.name):
                self.conv_3x3.build(None)
        if getattr(self, "reduce_1x1", None) is not None:
            with tf.name_scope(self.reduce_1x1.name):
                self.reduce_1x1.build(None)


class TFMobileViTMobileNetLayer(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_stages: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.layers = []
        for i in range(num_stages):
            layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
                name=f"layer.{i}",
            )
            self.layers.append(layer)
            in_channels = out_channels

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer_module in self.layers:
            features = layer_module(features, training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


class TFMobileViTSelfAttention(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)

        self.query = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="query")
        self.key = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="key")
        self.value = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="value")

        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.hidden_size = hidden_size

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(hidden_states)[0]

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, shape=(batch_size, -1, self.all_head_size))
        return context_layer

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.hidden_size])


class TFMobileViTSelfOutput(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(hidden_size, name="dense")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


class TFMobileViTAttention(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = TFMobileViTSelfAttention(config, hidden_size, name="attention")
        self.dense_output = TFMobileViTSelfOutput(config, hidden_size, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        self_outputs = self.attention(hidden_states, training=training)
        attention_output = self.dense_output(self_outputs, training=training)
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


class TFMobileViTIntermediate(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(intermediate_size, name="dense")
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hidden_size])


class TFMobileViTOutput(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(hidden_size, name="dense")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.intermediate_size = intermediate_size

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = hidden_states + input_tensor
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.intermediate_size])


class TFMobileViTTransformerLayer(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = TFMobileViTAttention(config, hidden_size, name="attention")
        self.intermediate = TFMobileViTIntermediate(config, hidden_size, intermediate_size, name="intermediate")
        self.mobilevit_output = TFMobileViTOutput(config, hidden_size, intermediate_size, name="output")
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states), training=training)
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.mobilevit_output(layer_output, hidden_states, training=training)
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
        if getattr(self, "mobilevit_output", None) is not None:
            with tf.name_scope(self.mobilevit_output.name):
                self.mobilevit_output.build(None)
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.hidden_size])
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.hidden_size])


class TFMobileViTTransformer(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.layers = []
        for i in range(num_stages):
            transformer_layer = TFMobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
                name=f"layer.{i}",
            )
            self.layers.append(transformer_layer)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


class TFMobileViTLayer(keras.layers.Layer):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        if stride == 2:
            self.downsampling_layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
                name="downsampling_layer",
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        self.conv_kxk = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="conv_kxk",
        )

        self.conv_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            name="conv_1x1",
        )

        self.transformer = TFMobileViTTransformer(
            config, hidden_size=hidden_size, num_stages=num_stages, name="transformer"
        )

        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

        self.conv_projection = TFMobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1, name="conv_projection"
        )

        self.fusion = TFMobileViTConvLayer(
            config,
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="fusion",
        )
        self.hidden_size = hidden_size

    def unfolding(self, features: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = tf.cast(patch_width * patch_height, "int32")

        batch_size = tf.shape(features)[0]
        orig_height = tf.shape(features)[1]
        orig_width = tf.shape(features)[2]
        channels = tf.shape(features)[3]

        new_height = tf.cast(tf.math.ceil(orig_height / patch_height) * patch_height, "int32")
        new_width = tf.cast(tf.math.ceil(orig_width / patch_width) * patch_width, "int32")

        interpolate = new_width != orig_width or new_height != orig_height
        if interpolate:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            features = tf.image.resize(features, size=(new_height, new_width), method="bilinear")

        # number of patches along width and height
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # convert from shape (batch_size, orig_height, orig_width, channels)
        # to the shape (batch_size * patch_area, num_patches, channels)
        features = tf.transpose(features, [0, 3, 1, 2])
        patches = tf.reshape(
            features, (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        )
        patches = tf.transpose(patches, [0, 2, 1, 3])
        patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
        patches = tf.transpose(patches, [0, 3, 2, 1])
        patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))

        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict

    def folding(self, patches: tf.Tensor, info_dict: Dict) -> tf.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))
        features = tf.transpose(features, perm=(0, 3, 2, 1))
        features = tf.reshape(
            features, (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
        )
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        features = tf.reshape(
            features, (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
        )
        features = tf.transpose(features, perm=(0, 2, 3, 1))

        if info_dict["interpolate"]:
            features = tf.image.resize(features, size=info_dict["orig_size"], method="bilinear")

        return features

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features, training=training)

        residual = features

        # local representation
        features = self.conv_kxk(features, training=training)
        features = self.conv_1x1(features, training=training)

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches, training=training)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features, training=training)
        features = self.fusion(tf.concat([residual, features], axis=-1), training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv_kxk", None) is not None:
            with tf.name_scope(self.conv_kxk.name):
                self.conv_kxk.build(None)
        if getattr(self, "conv_1x1", None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.hidden_size])
        if getattr(self, "conv_projection", None) is not None:
            with tf.name_scope(self.conv_projection.name):
                self.conv_projection.build(None)
        if getattr(self, "fusion", None) is not None:
            with tf.name_scope(self.fusion.name):
                self.fusion.build(None)
        if getattr(self, "downsampling_layer", None) is not None:
            with tf.name_scope(self.downsampling_layer.name):
                self.downsampling_layer.build(None)


class TFMobileViTEncoder(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.layers = []

        # segmentation architectures like DeepLab and PSPNet modify the strides
        # of the classification backbones
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        dilation = 1

        layer_1 = TFMobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
            name="layer.0",
        )
        self.layers.append(layer_1)

        layer_2 = TFMobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
            name="layer.1",
        )
        self.layers.append(layer_2)

        layer_3 = TFMobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[2],
            out_channels=config.neck_hidden_sizes[3],
            stride=2,
            hidden_size=config.hidden_sizes[0],
            num_stages=2,
            name="layer.2",
        )
        self.layers.append(layer_3)

        if dilate_layer_4:
            dilation *= 2

        layer_4 = TFMobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[3],
            out_channels=config.neck_hidden_sizes[4],
            stride=2,
            hidden_size=config.hidden_sizes[1],
            num_stages=4,
            dilation=dilation,
            name="layer.3",
        )
        self.layers.append(layer_4)

        if dilate_layer_5:
            dilation *= 2

        layer_5 = TFMobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[4],
            out_channels=config.neck_hidden_sizes[5],
            stride=2,
            hidden_size=config.hidden_sizes[2],
            num_stages=3,
            dilation=dilation,
            name="layer.4",
        )
        self.layers.append(layer_5)

    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, training=training)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)


@keras_serializable
class TFMobileViTMainLayer(keras.layers.Layer):
    config_class = MobileViTConfig

    def __init__(self, config: MobileViTConfig, expand_output: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.expand_output = expand_output

        self.conv_stem = TFMobileViTConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=config.neck_hidden_sizes[0],
            kernel_size=3,
            stride=2,
            name="conv_stem",
        )

        self.encoder = TFMobileViTEncoder(config, name="encoder")

        if self.expand_output:
            self.conv_1x1_exp = TFMobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5],
                out_channels=config.neck_hidden_sizes[6],
                kernel_size=1,
                name="conv_1x1_exp",
            )

        self.pooler = keras.layers.GlobalAveragePooling2D(data_format="channels_first", name="pooler")

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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        embedding_output = self.conv_stem(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # Change to NCHW output format to have uniformity in the modules
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])

            # global average pooling: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = self.pooler(last_hidden_state)
        else:
            last_hidden_state = encoder_outputs[0]
            # Change to NCHW output format to have uniformity in the modules
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])
            pooled_output = None

        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)

            # Change to NCHW output format to have uniformity in the modules
            if not self.expand_output:
                remaining_encoder_outputs = encoder_outputs[1:]
                remaining_encoder_outputs = tuple(
                    [tf.transpose(h, perm=(0, 3, 1, 2)) for h in remaining_encoder_outputs[0]]
                )
                remaining_encoder_outputs = (remaining_encoder_outputs,)
                return output + remaining_encoder_outputs
            else:
                return output + encoder_outputs[1:]

        # Change the other hidden state outputs to NCHW as well
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv_stem", None) is not None:
            with tf.name_scope(self.conv_stem.name):
                self.conv_stem.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build([None, None, None, None])
        if getattr(self, "conv_1x1_exp", None) is not None:
            with tf.name_scope(self.conv_1x1_exp.name):
                self.conv_1x1_exp.build(None)


class TFMobileViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileViTConfig
    base_model_prefix = "mobilevit"
    main_input_name = "pixel_values"


MOBILEVIT_START_DOCSTRING = r"""
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

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`MobileViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
"""


@add_start_docstrings(
    "The bare MobileViT model outputting raw hidden-states without any specific head on top.",
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTModel(TFMobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, expand_output: bool = True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.expand_output = expand_output

        self.mobilevit = TFMobileViTMainLayer(config, expand_output=expand_output, name="mobilevit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        output = self.mobilevit(pixel_values, output_hidden_states, return_dict, training=training)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)


@add_start_docstrings(
    """
    MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForImageClassification(TFMobileViTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: MobileViTConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.mobilevit = TFMobileViTMainLayer(config, name="mobilevit")

        # Classifier head
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = (
            keras.layers.Dense(config.num_labels, name="classifier") if config.num_labels > 0 else tf.identity
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[tuple, TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(self.dropout(pooled_output, training=training))
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.neck_hidden_sizes[-1]])


class TFMobileViTASPPPooling(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.global_pool = keras.layers.GlobalAveragePooling2D(keepdims=True, name="global_pool")

        self.conv_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            name="conv_1x1",
        )

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        spatial_size = shape_list(features)[1:-1]
        features = self.global_pool(features)
        features = self.conv_1x1(features, training=training)
        features = tf.image.resize(features, size=spatial_size, method="bilinear")
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "global_pool", None) is not None:
            with tf.name_scope(self.global_pool.name):
                self.global_pool.build([None, None, None, None])
        if getattr(self, "conv_1x1", None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)


class TFMobileViTASPP(keras.layers.Layer):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels

        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        self.convs = []

        in_projection = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
            name="convs.0",
        )
        self.convs.append(in_projection)

        self.convs.extend(
            [
                TFMobileViTConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                    name=f"convs.{i + 1}",
                )
                for i, rate in enumerate(config.atrous_rates)
            ]
        )

        pool_layer = TFMobileViTASPPPooling(
            config, in_channels, out_channels, name=f"convs.{len(config.atrous_rates) + 1}"
        )
        self.convs.append(pool_layer)

        self.project = TFMobileViTConvLayer(
            config,
            in_channels=5 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
            name="project",
        )

        self.dropout = keras.layers.Dropout(config.aspp_dropout_prob)

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        # since the hidden states were transposed to have `(batch_size, channels, height, width)`
        # layout we transpose them back to have `(batch_size, height, width, channels)` layout.
        features = tf.transpose(features, perm=[0, 2, 3, 1])
        pyramid = []
        for conv in self.convs:
            pyramid.append(conv(features, training=training))
        pyramid = tf.concat(pyramid, axis=-1)

        pooled_features = self.project(pyramid, training=training)
        pooled_features = self.dropout(pooled_features, training=training)
        return pooled_features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "project", None) is not None:
            with tf.name_scope(self.project.name):
                self.project.build(None)
        if getattr(self, "convs", None) is not None:
            for conv in self.convs:
                with tf.name_scope(conv.name):
                    conv.build(None)


class TFMobileViTDeepLabV3(keras.layers.Layer):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.aspp = TFMobileViTASPP(config, name="aspp")

        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)

        self.classifier = TFMobileViTConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
            name="classifier",
        )

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        features = self.aspp(hidden_states[-1], training=training)
        features = self.dropout(features, training=training)
        features = self.classifier(features, training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "aspp", None) is not None:
            with tf.name_scope(self.aspp.name):
                self.aspp.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)


@add_start_docstrings(
    """
    MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForSemanticSegmentation(TFMobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.num_labels = config.num_labels
        self.mobilevit = TFMobileViTMainLayer(config, expand_output=False, name="mobilevit")
        self.segmentation_head = TFMobileViTDeepLabV3(config, name="segmentation_head")

    def hf_compute_loss(self, logits, labels):
        # upsample logits to the images' original size
        # `labels` is of shape (batch_size, height, width)
        label_interp_shape = shape_list(labels)[1:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        # compute weighted loss
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        def masked_loss(real, pred):
            unmasked_loss = loss_fct(real, pred)
            mask = tf.cast(real != self.config.semantic_loss_ignore_index, dtype=unmasked_loss.dtype)
            masked_loss = unmasked_loss * mask
            # Reduction strategy in the similar spirit with
            # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_tf_utils.py#L210
            reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))

        return masked_loss(labels, upsampled_logits)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFSemanticSegmenterOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
        >>> model = TFMobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

        >>> inputs = image_processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and not self.config.num_labels > 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
            training=training,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.segmentation_head(encoder_hidden_states, training=training)

        loss = None
        if labels is not None:
            loss = self.hf_compute_loss(logits=logits, labels=labels)

        # make logits of shape (batch_size, num_labels, height, width) to
        # keep them consistent across APIs
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSemanticSegmenterOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilevit", None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        if getattr(self, "segmentation_head", None) is not None:
            with tf.name_scope(self.segmentation_head.name):
                self.segmentation_head.build(None)


__all__ = [
    "TFMobileViTForImageClassification",
    "TFMobileViTForSemanticSegmentation",
    "TFMobileViTModel",
    "TFMobileViTPreTrainedModel",
]
