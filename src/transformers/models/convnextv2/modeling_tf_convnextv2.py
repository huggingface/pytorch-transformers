# coding=utf-8
# Copyright 2023 Meta Platforms Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""TF 2.0 ConvNextV2 model."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPooling,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
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
from ...tf_utils import shape_list
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_convnextv2 import ConvNextV2Config


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnextv2-tiny-1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath with ConvNext->ConvNextV2
class TFConvNextV2DropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class TFConvNextV2GRN(keras.layers.Layer):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, config: ConvNextV2Config, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape: tf.TensorShape = None):
        # PT's `nn.Parameters` must be mapped to a TF layer weight to inherit the same name hierarchy (and vice-versa)
        self.weight = self.add_weight(
            name="weight",
            shape=(1, 1, 1, self.dim),
            initializer=keras.initializers.Zeros(),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, self.dim),
            initializer=keras.initializers.Zeros(),
        )
        return super().build(input_shape)

    def call(self, hidden_states: tf.Tensor):
        global_features = tf.norm(hidden_states, ord="euclidean", axis=(1, 2), keepdims=True)
        norm_features = global_features / (tf.reduce_mean(global_features, axis=-1, keepdims=True) + 1e-6)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        return hidden_states


# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextEmbeddings with ConvNext->ConvNextV2
class TFConvNextV2Embeddings(keras.layers.Layer):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        self.patch_embeddings = keras.layers.Conv2D(
            filters=config.hidden_sizes[0],
            kernel_size=config.patch_size,
            strides=config.patch_size,
            name="patch_embeddings",
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=keras.initializers.Zeros(),
        )
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        self.num_channels = config.num_channels
        self.config = config

    def call(self, pixel_values):
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            self.num_channels,
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build([None, None, None, self.config.num_channels])
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.config.hidden_sizes[0]])


class TFConvNextV2Layer(keras.layers.Layer):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch. Since we already permuted the inputs to follow
    NHWC ordering, we can just apply the operations straight-away without the permutation.

    Args:
        config (`ConvNextV2Config`):
            Model configuration class.
        dim (`int`):
            Number of input channels.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """

    def __init__(self, config: ConvNextV2Config, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.config = config
        self.dwconv = keras.layers.Conv2D(
            filters=dim,
            kernel_size=7,
            padding="same",
            groups=dim,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=keras.initializers.Zeros(),
            name="dwconv",
        )  # depthwise conv
        self.layernorm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="layernorm",
        )
        self.pwconv1 = keras.layers.Dense(
            units=4 * dim,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=keras.initializers.Zeros(),
            name="pwconv1",
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = get_tf_activation(config.hidden_act)
        self.grn = TFConvNextV2GRN(config, 4 * dim, dtype=tf.float32, name="grn")
        self.pwconv2 = keras.layers.Dense(
            units=dim,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=keras.initializers.Zeros(),
            name="pwconv2",
        )
        # Using `layers.Activation` instead of `tf.identity` to better control `training`
        # behaviour.
        self.drop_path = (
            TFConvNextV2DropPath(drop_path, name="drop_path")
            if drop_path > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )

    def call(self, hidden_states, training=False):
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x, training=training)
        x = input + x
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dwconv", None) is not None:
            with tf.name_scope(self.dwconv.name):
                self.dwconv.build([None, None, None, self.dim])
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.dim])
        if getattr(self, "pwconv1", None) is not None:
            with tf.name_scope(self.pwconv1.name):
                self.pwconv1.build([None, None, self.dim])
        if getattr(self, "grn", None) is not None:
            with tf.name_scope(self.grn.name):
                self.grn.build(None)
        if getattr(self, "pwconv2", None) is not None:
            with tf.name_scope(self.pwconv2.name):
                self.pwconv2.build([None, None, 4 * self.dim])
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)


# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextStage with ConvNext->ConvNextV2
class TFConvNextV2Stage(keras.layers.Layer):
    """ConvNextV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config (`ConvNextV2V2Config`):
            Model configuration class.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        depth (`int`):
            Number of residual blocks.
        drop_path_rates(`List[float]`):
            Stochastic depth rates for each layer.
    """

    def __init__(
        self,
        config: ConvNextV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        depth: int = 2,
        drop_path_rates: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = [
                keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    name="downsampling_layer.0",
                ),
                # Inputs to this layer will follow NHWC format since we
                # transposed the inputs from NCHW to NHWC in the `TFConvNextV2Embeddings`
                # layer. All the outputs throughout the model will be in NHWC
                # from this point on until the output where we again change to
                # NCHW.
                keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    kernel_initializer=get_initializer(config.initializer_range),
                    bias_initializer=keras.initializers.Zeros(),
                    name="downsampling_layer.1",
                ),
            ]
        else:
            self.downsampling_layer = [tf.identity]

        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = [
            TFConvNextV2Layer(
                config,
                dim=out_channels,
                drop_path=drop_path_rates[j],
                name=f"layers.{j}",
            )
            for j in range(depth)
        ]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def call(self, hidden_states):
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if self.in_channels != self.out_channels or self.stride > 1:
            with tf.name_scope(self.downsampling_layer[0].name):
                self.downsampling_layer[0].build([None, None, None, self.in_channels])
            with tf.name_scope(self.downsampling_layer[1].name):
                self.downsampling_layer[1].build([None, None, None, self.in_channels])


class TFConvNextV2Encoder(keras.layers.Layer):
    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        self.stages = []
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = TFConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
                name=f"stages.{i}",
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TFBaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    def build(self, input_shape=None):
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)


@keras_serializable
class TFConvNextV2MainLayer(keras.layers.Layer):
    config_class = ConvNextV2Config

    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embeddings = TFConvNextV2Embeddings(config, name="embeddings")
        self.encoder = TFConvNextV2Encoder(config, name="encoder")
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # We are setting the `data_format` like so because from here on we will revert to the
        # NCHW output format
        self.pooler = keras.layers.GlobalAvgPool2D(data_format="channels_last")

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        last_hidden_state = encoder_outputs[0]

        # Change to NCHW output format have uniformity in the modules
        pooled_output = self.pooler(last_hidden_state)
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        pooled_output = self.layernorm(pooled_output)

        # Change the other hidden state outputs to NCHW as well
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        if not return_dict:
            hidden_states = hidden_states if output_hidden_states else ()
            return (last_hidden_state, pooled_output) + hidden_states

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
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
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, self.config.hidden_sizes[-1]])


class TFConvNextV2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextV2Config
    base_model_prefix = "convnextv2"
    main_input_name = "pixel_values"


CONVNEXTV2_START_DOCSTRING = r"""
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
        config ([`ConvNextV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONVNEXTV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to `True`.
"""


@add_start_docstrings(
    "The bare ConvNextV2 model outputting raw features without any specific head on top.",
    CONVNEXTV2_START_DOCSTRING,
)
class TFConvNextV2Model(TFConvNextV2PreTrainedModel):
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.convnextv2 = TFConvNextV2MainLayer(config, name="convnextv2")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.convnextv2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return outputs[:]

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convnextv2", None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)


@add_start_docstrings(
    """
    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXTV2_START_DOCSTRING,
)
class TFConvNextV2ForImageClassification(TFConvNextV2PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.convnextv2 = TFConvNextV2MainLayer(config, name="convnextv2")

        # Classifier head
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=keras.initializers.Zeros(),
            name="classifier",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.convnextv2(
            pixel_values,
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

        return TFImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convnextv2", None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])


__all__ = ["TFConvNextV2ForImageClassification", "TFConvNextV2Model", "TFConvNextV2PreTrainedModel"]
