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
"""PyTorch Swin2SR Transformer model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_swin2sr import Swin2SRConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Swin2SRConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "caidas/swin2SR-classical-sr-x2-64"
_EXPECTED_OUTPUT_SHAPE = [1, 180, 488, 648]


@dataclass
class Swin2SREncoderOutput(ModelOutput):
    """
    Swin2SR encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


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


# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->Swin2SR
class Swin2SRDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Swin2SREmbeddings(nn.Module):
    """
    Construct the patch and optional position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = Swin2SRPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.window_size = config.window_size

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class Swin2SRPatchEmbeddings(nn.Module):
    def __init__(self, config, normalize_patches=True):
        super().__init__()
        num_channels = config.embed_dim
        image_size, patch_size = config.image_size, config.patch_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.projection = nn.Conv2d(num_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layernorm = nn.LayerNorm(config.embed_dim) if normalize_patches else None

    def forward(self, embeddings: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        embeddings = self.projection(embeddings)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings)

        return embeddings, output_dimensions


class Swin2SRPatchUnEmbeddings(nn.Module):
    r"""Image to Patch Unembedding"""

    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim

    def forward(self, embeddings, x_size):
        batch_size, height_width, num_channels = embeddings.shape
        embeddings = embeddings.transpose(1, 2).view(batch_size, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return embeddings


# Copied from transformers.models.swinv2.modeling_swinv2.Swinv2PatchMerging with Swinv2->Swin2SR
class Swin2SRPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # [batch_size, height/2 * width/2, 4*num_channels]
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # [batch_size, height/2 * width/2, 4*C]

        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature)

        return input_feature


# Copied from transformers.models.swinv2.modeling_swinv2.Swinv2SelfAttention with Swinv2->Swin2SR
class Swin2SRSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=[0, 0]):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        self.pretrained_window_size = pretrained_window_size
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        relative_coords_table = (
            torch.stack(meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # [1, 2*window_height - 1, 2*window_width - 1, 2]
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        elif window_size > 1:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        )
        # set to same dtype as mlp weight
        relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # cosine attention
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        # [window_height*window_width,window_height*window_width,num_attention_heads]
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        # [num_attention_heads,window_height*window_width,window_height*window_width]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swin2SRModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

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


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->Swin2SR
class Swin2SRSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.swinv2.modeling_swinv2.Swinv2Attention with Swinv2->Swin2SR
class Swin2SRAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        self.self = Swin2SRSelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        self.output = Swin2SRSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->Swin2SR
class Swin2SRIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput with Swin->Swin2SR
class Swin2SROutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.swinv2.modeling_swinv2.Swinv2Layer with Swinv2->Swin2SR
class Swin2SRLayer(nn.Module):
    def __init__(
        self, config, dim, input_resolution, num_heads, drop_path_rate=0.0, shift_size=0, pretrained_window_size=0
    ):
        super().__init__()
        self.input_resolution = input_resolution
        window_size, shift_size = self._compute_window_shift(
            (config.window_size, config.window_size), (shift_size, shift_size)
        )
        self.window_size = window_size[0]
        self.shift_size = shift_size[0]
        self.attention = Swin2SRAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swin2SRDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.intermediate = Swin2SRIntermediate(config, dim)
        self.output = Swin2SROutput(config, dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return window_size, shift_size

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        # pad hidden_states to multiples of window size
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.layernorm_before(attention_windows)
        hidden_states = shortcut + self.drop_path(hidden_states)

        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


class Swin2SRStage(nn.Module):
    """
    This corresponds to the Residual Swin Transformer Block (RSTB) in the original implementation.
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, pretrained_window_size=0):
        super().__init__()
        self.config = config
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                Swin2SRLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        if config.resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif config.resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = Swin2SRPatchEmbeddings(config, normalize_patches=False)

        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states

        height, width = input_dimensions
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

        output_dimensions = (height, width, height, width)

        hidden_states = self.patch_unembed(hidden_states, input_dimensions)
        hidden_states = self.conv(hidden_states)
        hidden_states, _ = self.patch_embed(hidden_states)

        hidden_states = hidden_states + residual

        stage_outputs = (hidden_states, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class Swin2SREncoder(nn.Module):
    def __init__(self, config, grid_size):
        super().__init__()
        self.num_stages = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.stages = nn.ModuleList(
            [
                Swin2SRStage(
                    config=config,
                    dim=config.embed_dim,
                    input_resolution=(grid_size[0], grid_size[1]),
                    depth=config.depths[stage_idx],
                    num_heads=config.num_heads[stage_idx],
                    drop_path=dpr[sum(config.depths[:stage_idx]) : sum(config.depths[: stage_idx + 1])],
                    pretrained_window_size=0,
                )
                for stage_idx in range(self.num_stages)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swin2SREncoderOutput]:
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        for i, stage_module in enumerate(self.stages):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions
                )
            else:
                layer_outputs = stage_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return Swin2SREncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Swin2SRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Swin2SRConfig
    base_model_prefix = "swin2sr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.trunc_normal_(module.weight.data, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SWIN2SR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Swin2SRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIN2SR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`Swin2SRImageProcessor.__call__`] for details.
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
    "The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN2SR_START_DOCSTRING,
)
class Swin2SRModel(Swin2SRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.num_channels == 3 and config.num_channels_out == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.img_range = config.img_range

        self.first_convolution = nn.Conv2d(config.num_channels, config.embed_dim, 3, 1, 1)
        self.embeddings = Swin2SREmbeddings(config)
        self.encoder = Swin2SREncoder(config, grid_size=self.embeddings.patch_embeddings.patches_resolution)

        self.layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)
        self.conv_after_body = nn.Conv2d(config.embed_dim, config.embed_dim, 3, 1, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def pad_and_normalize(self, pixel_values):
        _, _, height, width = pixel_values.size()

        # 1. pad
        window_size = self.config.window_size
        modulo_pad_height = (window_size - height % window_size) % window_size
        modulo_pad_width = (window_size - width % window_size) % window_size
        pixel_values = nn.functional.pad(pixel_values, (0, modulo_pad_width, 0, modulo_pad_height), "reflect")

        # 2. normalize
        self.mean = self.mean.type_as(pixel_values)
        pixel_values = (pixel_values - self.mean) * self.img_range

        return pixel_values

    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
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
    ) -> Union[Tuple, BaseModelOutput]:
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
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        _, _, height, width = pixel_values.shape

        # some preprocessing: padding + normalization
        pixel_values = self.pad_and_normalize(pixel_values)

        embeddings = self.first_convolution(pixel_values)
        embedding_output, input_dimensions = self.embeddings(embeddings)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        sequence_output = self.patch_unembed(sequence_output, (height, width))
        sequence_output = self.conv_after_body(sequence_output) + embeddings

        if not return_dict:
            output = (sequence_output,) + encoder_outputs[1:]

            return output

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Upsample(nn.Module):
    """Upsample module.

    Args:
        scale (`int`):
            Scale factor. Supported scales: 2^n and 3.
        num_features (`int`):
            Channel number of intermediate features.
    """

    def __init__(self, scale, num_features):
        super().__init__()

        self.scale = scale
        if (scale & (scale - 1)) == 0:
            # scale = 2^n
            for i in range(int(math.log(scale, 2))):
                self.add_module(f"convolution_{i}", nn.Conv2d(num_features, 4 * num_features, 3, 1, 1))
                self.add_module(f"pixelshuffle_{i}", nn.PixelShuffle(2))
        elif scale == 3:
            self.convolution = nn.Conv2d(num_features, 9 * num_features, 3, 1, 1)
            self.pixelshuffle = nn.PixelShuffle(3)
        else:
            raise ValueError(f"Scale {scale} is not supported. Supported scales: 2^n and 3.")

    def forward(self, hidden_state):
        if (self.scale & (self.scale - 1)) == 0:
            for i in range(int(math.log(self.scale, 2))):
                hidden_state = self.__getattr__(f"convolution_{i}")(hidden_state)
                hidden_state = self.__getattr__(f"pixelshuffle_{i}")(hidden_state)

        elif self.scale == 3:
            hidden_state = self.convolution(hidden_state)
            hidden_state = self.pixelshuffle(hidden_state)

        return hidden_state


class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)

    Used in lightweight SR to save parameters.

    Args:
        scale (int):
            Scale factor. Supported scales: 2^n and 3.
        in_channels (int):
            Channel number of intermediate features.
        out_channels (int):
            Channel number of output features.
    """

    def __init__(self, scale, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, (scale**2) * out_channels, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)

        return x


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.upsample = Upsample(config.upscale, num_features)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output):
        x = self.conv_before_upsample(sequence_output)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.final_convolution(x)

        return x


class NearestConvUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        if config.upscale != 4:
            raise ValueError("The nearest+conv upsampler only supports an upscale factor of 4 at the moment.")

        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, sequence_output):
        sequence_output = self.conv_before_upsample(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.lrelu(
            self.conv_up1(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        sequence_output = self.lrelu(
            self.conv_up2(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        reconstruction = self.final_convolution(self.lrelu(self.conv_hr(sequence_output)))
        return reconstruction


class PixelShuffleAuxUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()

        self.upscale = config.upscale
        self.conv_bicubic = nn.Conv2d(config.num_channels, num_features, 3, 1, 1)
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv_aux = nn.Conv2d(num_features, config.num_channels, 3, 1, 1)
        self.conv_after_aux = nn.Sequential(nn.Conv2d(3, num_features, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(config.upscale, num_features)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output, bicubic, height, width):
        bicubic = self.conv_bicubic(bicubic)
        sequence_output = self.conv_before_upsample(sequence_output)
        sequence_output = self.activation(sequence_output)
        aux = self.conv_aux(sequence_output)
        sequence_output = self.conv_after_aux(aux)
        sequence_output = (
            self.upsample(sequence_output)[:, :, : height * self.upscale, : width * self.upscale]
            + bicubic[:, :, : height * self.upscale, : width * self.upscale]
        )
        reconstruction = self.final_convolution(sequence_output)

        return reconstruction, aux


@add_start_docstrings(
    """
    Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.
    """,
    SWIN2SR_START_DOCSTRING,
)
class Swin2SRForImageSuperResolution(Swin2SRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.swin2sr = Swin2SRModel(config)
        self.upsampler = config.upsampler
        self.upscale = config.upscale

        # Upsampler
        num_features = 64
        if self.upsampler == "pixelshuffle":
            self.upsample = PixelShuffleUpsampler(config, num_features)
        elif self.upsampler == "pixelshuffle_aux":
            self.upsample = PixelShuffleAuxUpsampler(config, num_features)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(config.upscale, config.embed_dim, config.num_channels_out)
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            self.upsample = NearestConvUpsampler(config, num_features)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.final_convolution = nn.Conv2d(config.embed_dim, config.num_channels_out, 3, 1, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageSuperResolutionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageSuperResolutionOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> import numpy as np
         >>> from PIL import Image
         >>> import requests

         >>> from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

         >>> processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
         >>> model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

         >>> url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
         >>> image = Image.open(requests.get(url, stream=True).raw)
         >>> # prepare image for the model
         >>> inputs = processor(image, return_tensors="pt")

         >>> # forward pass
         >>> with torch.no_grad():
         ...     outputs = model(**inputs)

         >>> output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
         >>> output = np.moveaxis(output, source=0, destination=-1)
         >>> output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
         >>> # you can visualize `output` with `Image.fromarray`
         ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not supported at the moment")

        height, width = pixel_values.shape[2:]

        if self.config.upsampler == "pixelshuffle_aux":
            bicubic = nn.functional.interpolate(
                pixel_values,
                size=(height * self.upscale, width * self.upscale),
                mode="bicubic",
                align_corners=False,
            )

        outputs = self.swin2sr(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.upsampler in ["pixelshuffle", "pixelshuffledirect", "nearest+conv"]:
            reconstruction = self.upsample(sequence_output)
        elif self.upsampler == "pixelshuffle_aux":
            reconstruction, aux = self.upsample(sequence_output, bicubic, height, width)
            aux = aux / self.swin2sr.img_range + self.swin2sr.mean
        else:
            reconstruction = pixel_values + self.final_convolution(sequence_output)

        reconstruction = reconstruction / self.swin2sr.img_range + self.swin2sr.mean
        reconstruction = reconstruction[:, :, : height * self.upscale, : width * self.upscale]

        if not return_dict:
            output = (reconstruction,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageSuperResolutionOutput(
            loss=loss,
            reconstruction=reconstruction,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Swin2SRForImageSuperResolution", "Swin2SRModel", "Swin2SRPreTrainedModel"]
