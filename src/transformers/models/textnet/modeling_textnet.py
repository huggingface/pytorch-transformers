# coding=utf-8
# Copyright 2024 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch TextNet model."""

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import ACT2CLS
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from transformers.models.textnet.configuration_textnet import TextNetConfig
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.backbone_utils import BackboneMixin


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "TextNetConfig"
_CHECKPOINT_FOR_DOC = "czczup/textnet-base"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 20, 27]


class TextNetConvLayer(nn.Module):
    def __init__(self, config: TextNetConfig):
        super().__init__()

        self.kernel_size = config.stem_kernel_size
        self.stride = config.stem_stride
        self.activation_function = config.stem_act_func

        padding = (
            (config.kernel_size[0] // 2, config.kernel_size[1] // 2)
            if isinstance(config.stem_kernel_size, tuple)
            else config.stem_kernel_size // 2
        )

        self.conv = nn.Conv2d(
            config.stem_num_channels,
            config.stem_out_channels,
            kernel_size=config.stem_kernel_size,
            stride=config.stem_stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.stem_out_channels, config.batch_norm_eps)

        self.activation = nn.Identity()
        if self.activation_function is not None:
            self.activation = ACT2CLS[self.activation_function]()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        return self.activation(hidden_states)


class TextNetRepConvLayer(nn.Module):
    r"""
    This layer supports re-parameterization by combining multiple convolutional branches
    (e.g., main convolution, vertical, horizontal, and identity branches) during training.
    At inference time, these branches can be collapsed into a single convolution for
    efficiency, as per the re-parameterization paradigm.

    The "Rep" in the name stands for "re-parameterization" (introduced by RepVGG).
    """

    def __init__(self, config: TextNetConfig, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.num_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.activation_function = nn.ReLU()

        self.main_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.main_batch_norm = nn.BatchNorm2d(num_features=out_channels, eps=config.batch_norm_eps)

        vertical_padding = ((kernel_size[0] - 1) // 2, 0)
        horizontal_padding = (0, (kernel_size[1] - 1) // 2)

        if kernel_size[1] != 1:
            self.vertical_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=stride,
                padding=vertical_padding,
                bias=False,
            )
            self.vertical_batch_norm = nn.BatchNorm2d(num_features=out_channels, eps=config.batch_norm_eps)
        else:
            self.vertical_conv, self.vertical_batch_norm = None, None

        if kernel_size[0] != 1:
            self.horizontal_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size[1]),
                stride=stride,
                padding=horizontal_padding,
                bias=False,
            )
            self.horizontal_batch_norm = nn.BatchNorm2d(num_features=out_channels, eps=config.batch_norm_eps)
        else:
            self.horizontal_conv, self.horizontal_batch_norm = None, None

        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels, eps=config.batch_norm_eps)
            if out_channels == in_channels and stride == 1
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        main_outputs = self.main_conv(hidden_states)
        main_outputs = self.main_batch_norm(main_outputs)

        # applies a convolution with a vertical kernel
        if self.vertical_conv is not None:
            vertical_outputs = self.vertical_conv(hidden_states)
            vertical_outputs = self.vertical_batch_norm(vertical_outputs)
            main_outputs = main_outputs + vertical_outputs

        # applies a convolution with a horizontal kernel
        if self.horizontal_conv is not None:
            horizontal_outputs = self.horizontal_conv(hidden_states)
            horizontal_outputs = self.horizontal_batch_norm(horizontal_outputs)
            main_outputs = main_outputs + horizontal_outputs

        if self.rbr_identity is not None:
            id_out = self.rbr_identity(hidden_states)
            main_outputs = main_outputs + id_out

        return self.activation_function(main_outputs)


class TextNetStage(nn.Module):
    def __init__(self, config: TextNetConfig, depth: int):
        super().__init__()
        kernel_size = config.conv_layer_kernel_sizes[depth]
        stride = config.conv_layer_strides[depth]

        num_layers = len(kernel_size)
        stage_in_channel_size = config.hidden_sizes[depth]
        stage_out_channel_size = config.hidden_sizes[depth + 1]

        in_channels = [stage_in_channel_size] + [stage_out_channel_size] * (num_layers - 1)
        out_channels = [stage_out_channel_size] * num_layers

        stage = []
        for stage_config in zip(in_channels, out_channels, kernel_size, stride):
            stage.append(TextNetRepConvLayer(config, *stage_config))
        self.stage = nn.ModuleList(stage)

    def forward(self, hidden_state):
        for block in self.stage:
            hidden_state = block(hidden_state)
        return hidden_state


class TextNetEncoder(nn.Module):
    def __init__(self, config: TextNetConfig):
        super().__init__()

        stages = []
        num_stages = len(config.conv_layer_kernel_sizes)
        for stage_ix in range(num_stages):
            stages.append(TextNetStage(config, stage_ix))

        self.stages = nn.ModuleList(stages)

    def forward(
        self,
        hidden_state: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = [hidden_state]
        for stage in self.stages:
            hidden_state = stage(hidden_state)
            hidden_states.append(hidden_state)

        if not return_dict:
            output = (hidden_state,)
            return output + (hidden_states,) if output_hidden_states else output

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)


TEXTNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TextNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TEXTNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`TextNetImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TextNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TextNetConfig
    base_model_prefix = "textnet"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings(
    "The bare Textnet model outputting raw features without any specific head on top.",
    TEXTNET_START_DOCSTRING,
)
class TextNetModel(TextNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.stem = TextNetConvLayer(config)
        self.encoder = TextNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((2, 2))
        self.post_init()

    @add_start_docstrings_to_model_forward(TEXTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> Union[Tuple[Any, List[Any]], Tuple[Any], BaseModelOutputWithPoolingAndNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_state = self.stem(pixel_values)

        encoder_outputs = self.encoder(
            hidden_state, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            output = (last_hidden_state, pooled_output)
            return output + (encoder_outputs[1],) if output_hidden_states else output

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs[1] if output_hidden_states else None,
        )


@add_start_docstrings(
    """
    TextNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    TEXTNET_START_DOCSTRING,
)
class TextNetForImageClassification(TextNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.textnet = TextNetModel(config)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()

        # classification head
        self.classifier = nn.ModuleList([self.avg_pool, self.flatten])

        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TEXTNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> import torch
        >>> import requests
        >>> from transformers import TextNetForImageClassification, TextNetImageProcessor
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = TextNetImageProcessor.from_pretrained("czczup/textnet-base")
        >>> model = TextNetForImageClassification.from_pretrained("czczup/textnet-base")

        >>> inputs = processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> outputs.logits.shape
        torch.Size([1, 2])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.textnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = outputs[0]
        for layer in self.classifier:
            last_hidden_state = layer(last_hidden_state)
        logits = self.fc(last_hidden_state)
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
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


@add_start_docstrings(
    """
    TextNet backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    TEXTNET_START_DOCSTRING,
)
class TextNetBackbone(TextNetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.textnet = TextNetModel(config)
        self.num_features = config.hidden_sizes

        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TEXTNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> Union[Tuple[Tuple], BackboneOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoImageProcessor, AutoBackbone

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("czczup/textnet-base")
        >>> model = AutoBackbone.from_pretrained("czczup/textnet-base")

        >>> inputs = processor(image, return_tensors="pt")
        >>> with torch.no_grad():
        >>>     outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.textnet(pixel_values, output_hidden_states=True, return_dict=return_dict)

        hidden_states = outputs.hidden_states if return_dict else outputs[2]

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                hidden_states = outputs.hidden_states if return_dict else outputs[2]
                output += (hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )


__all__ = ["TextNetBackbone", "TextNetModel", "TextNetPreTrainedModel", "TextNetForImageClassification"]
