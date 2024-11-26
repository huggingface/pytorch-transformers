# coding=utf-8
# Copyright 2022 Intel Labs, OpenMMLab and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DPT (Dense Prediction Transformers) model.

This implementation is heavily inspired by OpenMMLab's implementation, found here:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/dpt_head.py.

"""

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging, torch_int
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DPTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "Intel/dpt-large"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]


@dataclass
class BaseModelOutputWithIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains intermediate activations that can be used at later stages. Useful
    in the context of Vision models.:

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """

    last_hidden_states: torch.FloatTensor = None
    intermediate_activations: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states as well as intermediate
    activations that can be used by the model at later stages.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    intermediate_activations: Optional[Tuple[torch.FloatTensor, ...]] = None


class DPTViTHybridEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, feature_size=None):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.backbone = load_backbone(config)
        feature_dim = self.backbone.channels[-1]
        if len(self.backbone.channels) != 3:
            raise ValueError(f"Expected backbone to have 3 output features, got {len(self.backbone.channels)}")
        self.residual_feature_map_index = [0, 1]  # Always take the output of the first and second backbone stage

        if feature_size is None:
            feat_map_shape = config.backbone_featmap_shape
            feature_size = feat_map_shape[-2:]
            feature_dim = feat_map_shape[1]
        else:
            feature_size = (
                feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            )
            feature_dim = self.backbone.channels[-1]

        self.image_size = image_size
        self.patch_size = patch_size[0]
        self.num_channels = num_channels

        self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]

        old_grid_size = torch_int(len(posemb_grid) ** 0.5)

        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, return_dict: bool = False
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // self.patch_size, width // self.patch_size
        )

        backbone_output = self.backbone(pixel_values)

        features = backbone_output.feature_maps[-1]

        # Retrieve also the intermediate activations to use them at later stages
        output_hidden_states = [backbone_output.feature_maps[index] for index in self.residual_feature_map_index]

        embeddings = self.projection(features).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + position_embeddings

        if not return_dict:
            return (embeddings, output_hidden_states)

        # Return hidden states and intermediate activations
        return BaseModelOutputWithIntermediateActivations(
            last_hidden_states=embeddings,
            intermediate_activations=output_hidden_states,
        )


class DPTViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = DPTViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]

        old_grid_size = torch_int(posemb_grid.size(0) ** 0.5)

        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(self, pixel_values, return_dict=False):
        batch_size, num_channels, height, width = pixel_values.shape

        # possibly interpolate position encodings to handle varying image sizes
        patch_size = self.config.patch_size
        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // patch_size, width // patch_size
        )

        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)

        if not return_dict:
            return (embeddings,)

        return BaseModelOutputWithIntermediateActivations(last_hidden_states=embeddings)


class DPTViTPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->DPT
class DPTViTSelfAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->DPT
class DPTViTSelfOutput(nn.Module):
    """
    The residual connection is defined in DPTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class DPTViTAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.attention = DPTViTSelfAttention(config)
        self.output = DPTViTSelfOutput(config)
        self.pruned_heads = set()

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.prune_heads
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->DPT
class DPTViTIntermediate(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->DPT
class DPTViTOutput(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# copied from transformers.models.vit.modeling_vit.ViTLayer with ViTConfig->DPTConfig, ViTAttention->DPTViTAttention, ViTIntermediate->DPTViTIntermediate, ViTOutput->DPTViTOutput
class DPTViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DPTViTAttention(config)
        self.intermediate = DPTViTIntermediate(config)
        self.output = DPTViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# copied from transformers.models.vit.modeling_vit.ViTEncoder with ViTConfig -> DPTConfig, ViTLayer->DPTViTLayer
class DPTViTEncoder(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([DPTViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class DPTReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList()
        if config.is_hybrid:
            self._init_reassemble_dpt_hybrid(config)
        else:
            self._init_reassemble_dpt(config)

        self.neck_ignore_stages = config.neck_ignore_stages

    def _init_reassemble_dpt_hybrid(self, config):
        r""" "
        For DPT-Hybrid the first 2 reassemble layers are set to `nn.Identity()`, please check the official
        implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
        for more details.
        """
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            if i <= 1:
                self.layers.append(nn.Identity())
            elif i > 1:
                self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type != "project":
            raise ValueError(f"Readout type {config.readout_type} is not supported for DPT-Hybrid.")

        # When using DPT-Hybrid the readout type is set to "project". The sanity check is done on the config file
        self.readout_projects = nn.ModuleList()
        hidden_size = _get_backbone_hidden_size(config)
        for i in range(len(config.neck_hidden_sizes)):
            if i <= 1:
                self.readout_projects.append(nn.Sequential(nn.Identity()))
            elif i > 1:
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def _init_reassemble_dpt(self, config):
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            hidden_size = _get_backbone_hidden_size(config)
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            if i not in self.neck_ignore_stages:
                # reshape to (batch_size, num_channels, height, width)
                cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
                batch_size, sequence_length, num_channels = hidden_state.shape
                if patch_height is not None and patch_width is not None:
                    hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
                else:
                    size = torch_int(sequence_length**0.5)
                    hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_shape = hidden_state.shape
                if self.config.readout_type == "project":
                    # reshape to (batch_size, height*width, num_channels)
                    hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
                    readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                    # concatenate the readout token to the hidden states and project
                    hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
                    # reshape back to (batch_size, num_channels, height, width)
                    hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
                elif self.config.readout_type == "add":
                    hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                    hidden_state = hidden_state.reshape(feature_shape)
                hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


def _get_backbone_hidden_size(config):
    if config.backbone_config is not None and config.is_hybrid is False:
        return config.backbone_config.hidden_size
    else:
        return config.hidden_size


class DPTReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor):
        super().__init__()
        # projection
        hidden_size = _get_backbone_hidden_size(config)
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)
        return hidden_state


class DPTFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DPTFeatureFusionLayer(config))

    def forward(self, hidden_states):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        fused_hidden_state = None
        for hidden_state, layer in zip(hidden_states, self.layers):
            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = layer(hidden_state)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DPTPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


class DPTFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """

    def __init__(self, config, align_corners=True):
        super().__init__()

        self.align_corners = align_corners

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        self.residual_layer1 = DPTPreActResidualLayer(config)
        self.residual_layer2 = DPTPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class DPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPTConfig
    base_model_prefix = "dpt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


DPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DPT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
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
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DPT Model transformer outputting raw hidden-states without any specific head on top.",
    DPT_START_DOCSTRING,
)
class DPTModel(DPTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # vit encoder
        if config.is_hybrid:
            self.embeddings = DPTViTHybridEmbeddings(config)
        else:
            self.embeddings = DPTViTEmbeddings(config)
        self.encoder = DPTViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = DPTViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        if self.config.is_hybrid:
            return self.embeddings
        else:
            return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndIntermediateActivations,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndIntermediateActivations]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, return_dict=return_dict)

        embedding_last_hidden_states = embedding_output[0] if not return_dict else embedding_output.last_hidden_states

        encoder_outputs = self.encoder(
            embedding_last_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:] + embedding_output[1:]

        return BaseModelOutputWithPoolingAndIntermediateActivations(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            intermediate_activations=embedding_output.intermediate_activations,
        )


# Copied from transformers.models.vit.modeling_vit.ViTPooler with ViT->DPT
class DPTViTPooler(nn.Module):
    def __init__(self, config: DPTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DPTNeck(nn.Module):
    """
    DPTNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DPT, it includes 2 stages:

    * DPTReassembleStage
    * DPTFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # postprocessing: only required in case of a non-hierarchical backbone (e.g. ViT, BEiT)
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = DPTReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # fusion
        self.fusion_stage = DPTFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output


class DPTDepthEstimationHead(nn.Module):
    """
    Output head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.projection = None
        if config.add_projection:
            self.projection = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        features = config.fusion_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)

        predicted_depth = self.head(hidden_states)

        predicted_depth = predicted_depth.squeeze(dim=1)

        return predicted_depth


@add_start_docstrings(
    """
    DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """,
    DPT_START_DOCSTRING,
)
class DPTForDepthEstimation(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = None
        if config.is_hybrid is False and (config.backbone_config is not None or config.backbone is not None):
            self.backbone = load_backbone(config)
        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)

        # Neck
        self.neck = DPTNeck(config)

        # Depth estimation head
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = image_processor.post_process_depth_estimation(
        ...     outputs,
        ...     target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 255 / predicted_depth.max()
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint8"))
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if self.backbone is not None:
            outputs = self.backbone.forward_with_filtered_kwargs(
                pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            hidden_states = outputs.feature_maps
        else:
            outputs = self.dpt(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )
            hidden_states = outputs.hidden_states if return_dict else outputs[1]
            # only keep certain features based on config.backbone_out_indices
            # note that the hidden_states also include the initial embeddings
            if not self.config.is_hybrid:
                hidden_states = [
                    feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
                ]
            else:
                backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
                backbone_hidden_states.extend(
                    feature
                    for idx, feature in enumerate(hidden_states[1:])
                    if idx in self.config.backbone_out_indices[2:]
                )

                hidden_states = backbone_hidden_states

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = pixel_values.shape
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states)

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class DPTSemanticSegmentationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        features = config.fusion_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Dropout(config.semantic_classifier_dropout),
            nn.Conv2d(features, config.num_labels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        logits = self.head(hidden_states)

        return logits


class DPTAuxiliaryHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        features = config.fusion_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, config.num_labels, kernel_size=1),
        )

    def forward(self, hidden_states):
        logits = self.head(hidden_states)

        return logits


@add_start_docstrings(
    """
    DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DPT_START_DOCSTRING,
)
class DPTForSemanticSegmentation(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dpt = DPTModel(config, add_pooling_layer=False)

        # Neck
        self.neck = DPTNeck(config)

        # Segmentation head(s)
        self.head = DPTSemanticSegmentationHead(config)
        self.auxiliary_head = DPTAuxiliaryHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large-ade")
        >>> model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.dpt(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features based on config.backbone_out_indices
        # note that the hidden_states also include the initial embeddings
        if not self.config.is_hybrid:
            hidden_states = [
                feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
            ]
        else:
            backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
            backbone_hidden_states.extend(
                feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices[2:]
            )

            hidden_states = backbone_hidden_states

        hidden_states = self.neck(hidden_states=hidden_states)

        logits = self.head(hidden_states)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(hidden_states[-1])

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if auxiliary_logits is not None:
                upsampled_auxiliary_logits = nn.functional.interpolate(
                    auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
            # compute weighted loss
            loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            main_loss = loss_fct(upsampled_logits, labels)
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
