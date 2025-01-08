# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""TensorFlow RegNet model."""

from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "RegNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/regnet-y-040"
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/regnet-y-040"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


class TFRegNetConvLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The padding and conv has been verified in
        # https://colab.research.google.com/gist/sayakpaul/854bc10eeaf21c9ee2119e0b9f3841a7/scratchpad.ipynb
        self.padding = keras.layers.ZeroPadding2D(padding=kernel_size // 2)
        self.convolution = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            groups=groups,
            use_bias=False,
            name="convolution",
        )
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.activation = ACT2FN[activation] if activation is not None else tf.identity
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, hidden_state):
        hidden_state = self.convolution(self.padding(hidden_state))
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])


class TFRegNetEmbeddings(keras.layers.Layer):
    """
    RegNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: RegNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = config.num_channels
        self.embedder = TFRegNetConvLayer(
            in_channels=config.num_channels,
            out_channels=config.embedding_size,
            kernel_size=3,
            stride=2,
            activation=config.hidden_act,
            name="embedder",
        )

    def call(self, pixel_values):
        num_channels = shape_list(pixel_values)[1]
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        hidden_state = self.embedder(pixel_values)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)


class TFRegNetShortCut(keras.layers.Layer):
    """
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.convolution = keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.normalization(self.convolution(inputs), training=training)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "convolution", None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, "normalization", None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])


class TFRegNetSELayer(keras.layers.Layer):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, in_channels: int, reduced_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True, name="pooler")
        self.attention = [
            keras.layers.Conv2D(filters=reduced_channels, kernel_size=1, activation="relu", name="attention.0"),
            keras.layers.Conv2D(filters=in_channels, kernel_size=1, activation="sigmoid", name="attention.2"),
        ]
        self.in_channels = in_channels
        self.reduced_channels = reduced_channels

    def call(self, hidden_state):
        # [batch_size, h, w, num_channels] -> [batch_size, 1, 1, num_channels]
        pooled = self.pooler(hidden_state)
        for layer_module in self.attention:
            pooled = layer_module(pooled)
        hidden_state = hidden_state * pooled
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention[0].name):
                self.attention[0].build([None, None, None, self.in_channels])
            with tf.name_scope(self.attention[1].name):
                self.attention[1].build([None, None, None, self.reduced_channels])


class TFRegNetXLayer(keras.layers.Layer):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = (
            TFRegNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else keras.layers.Activation("linear", name="shortcut")
        )
        # `self.layers` instead of `self.layer` because that is a reserved argument.
        self.layers = [
            TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name="layer.0"),
            TFRegNetConvLayer(
                out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act, name="layer.1"
            ),
            TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name="layer.2"),
        ]
        self.activation = ACT2FN[config.hidden_act]

    def call(self, hidden_state):
        residual = hidden_state
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetYLayer(keras.layers.Layer):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = (
            TFRegNetShortCut(in_channels, out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else keras.layers.Activation("linear", name="shortcut")
        )
        self.layers = [
            TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name="layer.0"),
            TFRegNetConvLayer(
                out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act, name="layer.1"
            ),
            TFRegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4)), name="layer.2"),
            TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name="layer.3"),
        ]
        self.activation = ACT2FN[config.hidden_act]

    def call(self, hidden_state):
        residual = hidden_state
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "shortcut", None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetStage(keras.layers.Layer):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(
        self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ):
        super().__init__(**kwargs)

        layer = TFRegNetXLayer if config.layer_type == "x" else TFRegNetYLayer
        self.layers = [
            # downsampling is done in the first layer with stride of 2
            layer(config, in_channels, out_channels, stride=stride, name="layers.0"),
            *[layer(config, out_channels, out_channels, name=f"layers.{i+1}") for i in range(depth - 1)],
        ]

    def call(self, hidden_state):
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFRegNetEncoder(keras.layers.Layer):
    def __init__(self, config: RegNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.stages = []
        # based on `downsample_in_first_stage`, the first layer of the first stage may or may not downsample the input
        self.stages.append(
            TFRegNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name="stages.0",
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, config.depths[1:])):
            self.stages.append(TFRegNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i+1}"))

    def call(
        self, hidden_state: tf.Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)


@keras_serializable
class TFRegNetMainLayer(keras.layers.Layer):
    config_class = RegNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedder = TFRegNetEmbeddings(config, name="embedder")
        self.encoder = TFRegNetEncoder(config, name="encoder")
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True, name="pooler")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> TFBaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)

        # Change to NCHW output format have uniformity in the modules
        pooled_output = tf.transpose(pooled_output, perm=(0, 3, 1, 2))
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))

        # Change the other hidden state outputs to NCHW as well
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedder", None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))


class TFRegNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RegNetConfig
    base_model_prefix = "regnet"
    main_input_name = "pixel_values"

    @property
    def input_signature(self):
        return {"pixel_values": tf.TensorSpec(shape=(None, self.config.num_channels, 224, 224), dtype=tf.float32)}


REGNET_START_DOCSTRING = r"""
    This model is a Tensorflow
    [keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

REGNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConveNextImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RegNet model outputting raw features without any specific head on top.",
    REGNET_START_DOCSTRING,
)
class TFRegNetModel(TFRegNetPreTrainedModel):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.regnet = TFRegNetMainLayer(config, name="regnet")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.regnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        if not return_dict:
            return (outputs[0],) + outputs[1:]

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "regnet", None) is not None:
            with tf.name_scope(self.regnet.name):
                self.regnet.build(None)


@add_start_docstrings(
    """
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    REGNET_START_DOCSTRING,
)
class TFRegNetForImageClassification(TFRegNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.regnet = TFRegNetMainLayer(config, name="regnet")
        # classification head
        self.classifier = [
            keras.layers.Flatten(),
            keras.layers.Dense(config.num_labels, name="classifier.1") if config.num_labels > 0 else tf.identity,
        ]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.regnet(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        flattened_output = self.classifier[0](pooled_output)
        logits = self.classifier[1](flattened_output)

        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "regnet", None) is not None:
            with tf.name_scope(self.regnet.name):
                self.regnet.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier[1].name):
                self.classifier[1].build([None, None, None, self.config.hidden_sizes[-1]])


__all__ = ["TFRegNetForImageClassification", "TFRegNetModel", "TFRegNetPreTrainedModel"]
