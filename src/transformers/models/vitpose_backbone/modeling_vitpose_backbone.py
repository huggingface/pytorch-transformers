# coding=utf-8
# Copyright 2024 University of Sydney and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch VitPose backbone model.

This code is the same as the original Vision Transformer (ViT) with 2 modifications:
- use of padding=2 in the patch embedding layer
- addition of a mixture-of-experts MLP layer
"""

import collections.abc
import math
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitpose_backbone import VitPoseBackboneConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "VitPoseBackboneConfig"


class VitPoseBackbonePatchEmbeddings(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        embed_dim = config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values)

        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class VitPoseBackboneEmbeddings(nn.Module):
    """
    Construct the position and patch embeddings.
    """

    def __init__(self, config: VitPoseBackboneConfig) -> None:
        super().__init__()

        self.patch_embeddings = VitPoseBackbonePatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings[:, 1:] + self.position_embeddings[:, :1]

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->VitPoseBackbone
class VitPoseBackboneSelfAttention(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig) -> None:
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


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->VitPoseBackbone
class VitPoseBackboneSelfOutput(nn.Module):
    """
    The residual connection is defined in VitPoseBackboneLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VitPoseBackboneConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->VitPoseBackbone
class VitPoseBackboneAttention(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig) -> None:
        super().__init__()
        self.attention = VitPoseBackboneSelfAttention(config)
        self.output = VitPoseBackboneSelfOutput(config)
        self.pruned_heads = set()

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


class VitPoseBackboneMoeMLP(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__()

        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)

        num_experts = config.num_experts
        part_features = config.part_features

        self.part_features = part_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features - part_features)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

        self.num_experts = num_experts
        experts = [nn.Linear(hidden_features, part_features) for _ in range(num_experts)]
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expert_hidden_state = torch.zeros_like(hidden_state[:, :, -self.part_features :])

        hidden_state = self.fc1(hidden_state)
        hidden_state = self.act(hidden_state)
        shared_hidden_state = self.fc2(hidden_state)
        indices = indices.view(-1, 1, 1)

        # to support ddp training
        for i in range(self.num_experts):
            selected_index = indices == i
            current_hidden_state = self.experts[i](hidden_state) * selected_index
            expert_hidden_state = expert_hidden_state + current_hidden_state

        hidden_state = torch.cat([shared_hidden_state, expert_hidden_state], dim=-1)

        return hidden_state


class VitPoseBackboneMLP(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VitPoseBackboneLayer(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.attention = VitPoseBackboneAttention(config)
        self.mlp = VitPoseBackboneMLP(config) if self.num_experts == 1 else VitPoseBackboneMoeMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # Validate dataset_index when using multiple experts
        if self.num_experts > 1 and dataset_index is None:
            raise ValueError(
                "dataset_index must be provided when using multiple experts "
                f"(num_experts={self.num_experts}). Please provide dataset_index "
                "to the forward pass."
            )
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in VitPoseBackbone, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        if self.num_experts == 1:
            layer_output = self.mlp(layer_output)
        else:
            layer_output = self.mlp(layer_output, indices=dataset_index)

        # second residual connection
        layer_output = layer_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VitPoseBackbone
class VitPoseBackboneEncoder(nn.Module):
    def __init__(self, config: VitPoseBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VitPoseBackboneLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
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
                    dataset_index,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, dataset_index, layer_head_mask, output_attentions)

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


class VitPoseBackbonePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitPoseBackboneConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VitPoseBackboneEmbeddings", "VitPoseBackboneLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VitPoseBackboneEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


VITPOSE_BACKBONE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VitPoseBackboneConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITPOSE_BACKBONE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.

        dataset_index (`torch.Tensor` of shape `(batch_size,)`):
            Index to use in the Mixture-of-Experts (MoE) blocks of the backbone.

            This corresponds to the dataset index used during training, e.g. index 0 refers to COCO.

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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The VitPose backbone useful for downstream tasks.",
    VITPOSE_BACKBONE_START_DOCSTRING,
)
class VitPoseBackbone(VitPoseBackbonePreTrainedModel, BackboneMixin):
    def __init__(self, config: VitPoseBackboneConfig):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = VitPoseBackboneEmbeddings(config)
        self.encoder = VitPoseBackboneEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VITPOSE_BACKBONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import VitPoseBackboneConfig, VitPoseBackbone
        >>> import torch

        >>> config = VitPoseBackboneConfig(out_indices=[-1])
        >>> model = VitPoseBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 256, 192)
        >>> dataset_index = torch.tensor([1])
        >>> outputs = model(pixel_values, dataset_index)
        ```"""
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

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            dataset_index=dataset_index,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.layernorm(hidden_state)
                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
