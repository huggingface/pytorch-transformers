# coding=utf-8
# Copyright 2022 Sea AI Lab and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PoolFormer model."""

import collections.abc
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_poolformer import PoolFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "PoolFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "sail/poolformer_s12"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "sail/poolformer_s12"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->PoolFormer
class PoolFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class PoolFormerEmbeddings(nn.Module):
    """
    Construct Patch Embeddings.
    """

    def __init__(self, hidden_size, num_channels, patch_size, stride, padding, norm_layer=None):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(hidden_size) if norm_layer else nn.Identity()

    def forward(self, pixel_values):
        embeddings = self.projection(pixel_values)
        embeddings = self.norm(embeddings)
        return embeddings


class PoolFormerGroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group. Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PoolFormerPooling(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, hidden_states):
        return self.pool(hidden_states) - hidden_states


class PoolFormerOutput(nn.Module):
    def __init__(self, config, dropout_prob, hidden_size, intermediate_size):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, intermediate_size, 1)
        self.conv2 = nn.Conv2d(intermediate_size, hidden_size, 1)
        self.drop = PoolFormerDropPath(dropout_prob)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.drop(hidden_states)

        return hidden_states


class PoolFormerLayer(nn.Module):
    """This corresponds to the 'PoolFormerBlock' class in the original implementation."""

    def __init__(self, config, num_channels, pool_size, hidden_size, intermediate_size, drop_path):
        super().__init__()
        self.pooling = PoolFormerPooling(pool_size)
        self.output = PoolFormerOutput(config, drop_path, hidden_size, intermediate_size)
        self.before_norm = PoolFormerGroupNorm(num_channels)
        self.after_norm = PoolFormerGroupNorm(num_channels)

        # Useful for training neural nets
        self.drop_path = PoolFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((num_channels)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((num_channels)), requires_grad=True
            )

    def forward(self, hidden_states):
        if self.use_layer_scale:
            pooling_output = self.pooling(self.before_norm(hidden_states))
            scaled_op = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * pooling_output
            # First residual connection
            hidden_states = hidden_states + self.drop_path(scaled_op)
            outputs = ()

            layer_output = self.output(self.after_norm(hidden_states))
            scaled_op = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * layer_output
            # Second residual connection
            output = hidden_states + self.drop_path(scaled_op)

            outputs = (output,) + outputs
            return outputs

        else:
            pooling_output = self.drop_path(self.pooling(self.before_norm(hidden_states)))
            # First residual connection
            hidden_states = pooling_output + hidden_states
            outputs = ()

            # Second residual connection inside the PoolFormerOutput block
            layer_output = self.drop_path(self.output(self.after_norm(hidden_states)))
            output = hidden_states + layer_output

            outputs = (output,) + outputs
            return outputs


class PoolFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                PoolFormerEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    padding=config.padding[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    PoolFormerLayer(
                        config,
                        num_channels=config.hidden_sizes[i],
                        pool_size=config.pool_size,
                        hidden_size=config.hidden_sizes[i],
                        intermediate_size=int(config.hidden_sizes[i] * config.mlp_ratio),
                        drop_path=dpr[cur + j],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        hidden_states = pixel_values
        for idx, layers in enumerate(zip(self.patch_embeddings, self.block)):
            embedding_layer, block_layer = layers
            # Get patch embeddings from hidden_states
            hidden_states = embedding_layer(hidden_states)
            # Send the embeddings through the blocks
            for _, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states)
                hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class PoolFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PoolFormerConfig
    base_model_prefix = "poolformer"
    main_input_name = "pixel_values"
    _no_split_modules = ["PoolFormerLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


POOLFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PoolFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

POOLFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`PoolFormerImageProcessor.__call__`] for details.
"""


@add_start_docstrings(
    "The bare PoolFormer Model transformer outputting raw hidden-states without any specific head on top.",
    POOLFORMER_START_DOCSTRING,
)
class PoolFormerModel(PoolFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = PoolFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output, None) + encoder_outputs[1:]

        return BaseModelOutputWithNoAttention(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class PoolFormerFinalPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        return output


@add_start_docstrings(
    """
    PoolFormer Model transformer with an image classification head on top
    """,
    POOLFORMER_START_DOCSTRING,
)
class PoolFormerForImageClassification(PoolFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.poolformer = PoolFormerModel(config)

        # Final norm
        self.norm = PoolFormerGroupNorm(config.hidden_sizes[-1])
        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.poolformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(self.norm(sequence_output).mean([-2, -1]))

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

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


__all__ = ["PoolFormerForImageClassification", "PoolFormerModel", "PoolFormerPreTrainedModel"]
