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
"""PyTorch MobileNetV1 model."""

from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileNetV1Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/mobilenet_v1_1.0_224"
_EXPECTED_OUTPUT_SHAPE = [1, 1024, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/mobilenet_v1_1.0_224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """

    tf_to_pt_map = {}

    if isinstance(model, MobileNetV1ForImageClassification):
        backbone = model.mobilenet_v1
    else:
        backbone = model

    prefix = "MobilenetV1/Conv2d_0/"
    tf_to_pt_map[prefix + "weights"] = backbone.conv_stem.convolution.weight
    tf_to_pt_map[prefix + "BatchNorm/beta"] = backbone.conv_stem.normalization.bias
    tf_to_pt_map[prefix + "BatchNorm/gamma"] = backbone.conv_stem.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.normalization.running_var

    for i in range(13):
        tf_index = i + 1
        pt_index = i * 2

        pointer = backbone.layer[pt_index]
        prefix = f"MobilenetV1/Conv2d_{tf_index}_depthwise/"
        tf_to_pt_map[prefix + "depthwise_weights"] = pointer.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = pointer.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = pointer.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.normalization.running_var

        pointer = backbone.layer[pt_index + 1]
        prefix = f"MobilenetV1/Conv2d_{tf_index}_pointwise/"
        tf_to_pt_map[prefix + "weights"] = pointer.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = pointer.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = pointer.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.normalization.running_var

    if isinstance(model, MobileNetV1ForImageClassification):
        prefix = "MobilenetV1/Logits/Conv2d_1c_1x1/"
        tf_to_pt_map[prefix + "weights"] = model.classifier.weight
        tf_to_pt_map[prefix + "biases"] = model.classifier.bias

    return tf_to_pt_map


def load_tf_weights_in_mobilenet_v1(model, config, tf_checkpoint_path):
    """Load TensorFlow checkpoints in a PyTorch model."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        array = tf_weights[name]

        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")
            array = np.transpose(array, (2, 3, 0, 1))
        elif "weights" in name:
            logger.info("Transposing")
            if len(pointer.shape) == 2:  # copying into linear layer
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        logger.info(f"Initialize PyTorch weight {name} {array.shape}")
        pointer.data = torch.from_numpy(array)

        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor:
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    in_height, in_width = features.shape[-2:]
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size

    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return nn.functional.pad(features, padding, "constant", 0.0)


class MobileNetV1ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileNetV1Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        groups: Optional[int] = 1,
        bias: bool = False,
        use_normalization: Optional[bool] = True,
        use_activation: Optional[bool or str] = True,
    ) -> None:
        super().__init__()
        self.config = config

        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2)

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=config.layer_norm_eps,
                momentum=0.9997,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


class MobileNetV1PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileNetV1Config
    load_tf_weights = load_tf_weights_in_mobilenet_v1
    base_model_prefix = "mobilenet_v1"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MOBILENET_V1_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileNetV1Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILENET_V1_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileNetV1ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileNetV1 model outputting raw hidden-states without any specific head on top.",
    MOBILENET_V1_START_DOCSTRING,
)
class MobileNetV1Model(MobileNetV1PreTrainedModel):
    def __init__(self, config: MobileNetV1Config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        depth = 32
        out_channels = max(int(depth * config.depth_multiplier), config.min_depth)

        self.conv_stem = MobileNetV1ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
        )

        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        self.layer = nn.ModuleList()
        for i in range(13):
            in_channels = out_channels

            if strides[i] == 2 or i == 0:
                depth *= 2
                out_channels = max(int(depth * config.depth_multiplier), config.min_depth)

            self.layer.append(
                MobileNetV1ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=strides[i],
                    groups=in_channels,
                )
            )

            self.layer.append(
                MobileNetV1ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                )
            )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @add_start_docstrings_to_model_forward(MOBILENET_V1_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.conv_stem(pixel_values)

        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        last_hidden_state = hidden_states

        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None

        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    MobileNetV1 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILENET_V1_START_DOCSTRING,
)
class MobileNetV1ForImageClassification(MobileNetV1PreTrainedModel):
    def __init__(self, config: MobileNetV1Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilenet_v1 = MobileNetV1Model(config)

        last_hidden_size = self.mobilenet_v1.layer[-1].convolution.out_channels

        # Classifier head
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = nn.Linear(last_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILENET_V1_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilenet_v1(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "MobileNetV1ForImageClassification",
    "MobileNetV1Model",
    "MobileNetV1PreTrainedModel",
    "load_tf_weights_in_mobilenet_v1",
]
