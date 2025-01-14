# coding=utf-8
# Copyright 2024 Transformers are Sample-Efficient World Models(IRIS) paper authors and The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Iris model."""

import math
import sys
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm import tqdm

from ...activations import ACT2FN
from ...cache_utils import StaticCache
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchvision_available,
    logging,
    replace_return_docstrings,
)
from .configuration_iris import IrisConfig


if is_torchvision_available():
    from torchvision import models

Batch = Dict[str, torch.Tensor]

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "ruffy369/iris-breakout"
_CONFIG_FOR_DOC = "IrisConfig"


@dataclass
class IrisOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        reconstructed_img (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed image from input frame(only one real image per batch is given as rest are in imagination of the world model) with discrete_autoencoder
        losses (`[torch.FloatTensor]` of shape `(3,)`):
            RL Agent's components' total loss [discrete_autoencoder, world model and actor critic].
        action_preds (logits) (`torch.FloatTensor` of shape `(batch_size, 1, num_actions)`):
            Policy action predictions with actor critic
        reward_preds (logits) (`torch.FloatTensor` of shape `(batch_size, sequence_length_world_model, 3)`):
            Predicted rewards for each state with world model
        epsiode_end (logits) (`torch.FloatTensor` of shape `(batch_size, sequence_length_world_model, 2)`):
            Predicted potential episode termination with world model
        obs_preds (logits) (`torch.FloatTensor` of shape `(batch_size, 320, vocab_size)`):
            Predicted tokens for next frame with world model
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of various shapes as there are three components in rl_agent and the shapes are:
            `(batch_size, resolution, resolution,resolution), (batch_size, resolution, resolution//2,resolution//2),(batch_size, resolution, resolution//4,resolution//4),
            (batch_size, resolution, resolution//8,resolution//8)(batch_size, resolution, resolution//16,resolution//16)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer)(len of tuple:22) of various shapes as there are three components in rl_agent and the shapes are:
            `(batch_size, embed_dim, embed_dim),(batch_size,resolution,resolution),(batch_size,attn_resolution,attn_resolution),(batch_size,num_heads,340,340)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    reconstructed_img: torch.FloatTensor = None
    losses: Tuple[torch.FloatTensor] = None
    action_preds: torch.FloatTensor = None
    reward_preds: torch.FloatTensor = None
    epsiode_end: torch.FloatTensor = None
    obs_preds: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class IrisPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = IrisConfig
    base_model_prefix = "iris"
    main_input_name = "observations"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, IrisModel):
            module.gradient_checkpointing = value


IRIS_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~IrisConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


IRIS_INPUTS_DOCSTRING = r"""
    Args:
        observations (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels, height, width)`):
            The image observations from real environment in L sequence length(timesteps) (only one frame is used from the environment and
            rest are imagined in the world model for training)
        actions (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The actions from real environment in L sequence length(timesteps)
        rewards (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The rewards from real environment in L sequence length(timesteps)
        ends (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
            The episode termination booleans from real environment in L sequence length(timesteps)
        mask_padding (`torch.BoolTensor` of shape `(batch_size, 1)`):
            The padding mask for input observations before putting them in encode and decoder
        should_preprocess (`bool`, *optional*):
            If set to `True`, `observations` are preprocessed before being passed to the model when encoding.
        should_postprocess(`bool`, *optional*):
            If set to `True`, `reconstructions` are postprocessed after being passed to the model when decoding.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.


"""


class IrisLossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class IrisSlicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        self.block_size = block_mask.size(0)
        self.num_kept_tokens = block_mask.sum().long().item()
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer("indices", kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[: num_blocks * self.num_kept_tokens]
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class IrisHead(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__()
        self.slicer = IrisSlicer(max_blocks, block_mask)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(
        self, x: torch.Tensor, output_hidden_states: bool = False, num_steps: int = None, prev_steps: int = None
    ) -> torch.Tensor:
        hidden_states = () if output_hidden_states else None
        x_sliced = x[
            :, self.slicer.compute_slice(num_steps, prev_steps)
        ]  # x is (batch_size, num_timesteps, tokens_per_frame)
        # Add the hidden state to the tuple after Relu activation
        x_sliced_hidden = self.head_module[1](self.head_module[0](x_sliced))
        hidden_states = hidden_states + (x_sliced_hidden,) if output_hidden_states else None
        hidden_states = hidden_states + (self.head_module[2](x_sliced_hidden),) if output_hidden_states else None

        return self.head_module(x_sliced), hidden_states


class IrisEmbedder(nn.Module):
    def __init__(self, max_blocks: int, block_masks: List[torch.Tensor], embedding_tables: List[nn.Embedding]) -> None:
        super().__init__()
        assert len(block_masks) == len(embedding_tables)
        assert (sum(block_masks) == 1).all()  # block mask are a partition of a block
        self.embedding_dim = embedding_tables[0].embedding_dim
        assert all(e.embedding_dim == self.embedding_dim for e in embedding_tables)
        self.embedding_tables = embedding_tables
        self.slicers = [IrisSlicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        assert tokens.ndim == 2  # x is (batch_size, num_timesteps)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            output[:, s] = emb(tokens[:, s])
        return output


class IrisEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.ch_mult)
        timestep_embedding_channels = 0  # timestep embedding #channels

        # downsampling
        self.conv_in = nn.Conv2d(config.in_channels, config.ch, kernel_size=3, stride=1, padding=1)

        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.config.num_res_blocks):
                block.append(
                    IrisResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        timestep_embedding_channels=timestep_embedding_channels,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(IrisAttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = IrisDownsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = IrisResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=timestep_embedding_channels,
            dropout=config.dropout,
        )
        self.mid.attn_1 = IrisAttnBlock(block_in)
        self.mid.block_2 = IrisResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=timestep_embedding_channels,
            dropout=config.dropout,
        )

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, config.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self, x: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        timestep_embedding = None  # timestep embedding

        # downsampling
        hs = [self.conv_in(x)]

        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        hidden_states = hidden_states + (hs[0],) if output_hidden_states else None

        for i_level in range(self.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], timestep_embedding)
                if len(self.down[i_level].attn) > 0:
                    attn_out = self.down[i_level].attn[i_block](h)
                    h = attn_out[0]
                    # Add attention weights
                    attentions = attentions + (attn_out[1],) if output_attentions else None
                hs.append(h)
                # Add the hidden state to the tuple
                hidden_states = hidden_states + (h,) if output_hidden_states else None
            if i_level != self.num_resolutions - 1:
                hs_dummy = self.down[i_level].downsample(hs[-1])
                hs.append(hs_dummy)
                hidden_states = hidden_states + (hs_dummy,) if output_hidden_states else None

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, timestep_embedding)
        # Add the hidden state to the tuple
        hidden_states = hidden_states + (h,) if output_hidden_states else None
        attn_out = self.mid.attn_1(h)
        h = attn_out[0]
        # Add attention weights
        attentions = attentions + (attn_out[1],) if output_attentions else None
        # Add the hidden state to the tuple
        hidden_states = hidden_states + (h,) if output_hidden_states else None
        h = self.mid.block_2(h, timestep_embedding)
        # Add the hidden state to the tuple
        hidden_states = hidden_states + (h,) if output_hidden_states else None

        # end
        h = self.norm_out(h)
        h = ACT2FN["swish"](h)
        h = self.conv_out(h)

        return h, hidden_states, attentions


class IrisDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        timestep_embedding_channels = 0
        self.num_resolutions = len(config.ch_mult)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = nn.Conv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = IrisResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=timestep_embedding_channels,
            dropout=config.dropout,
        )
        self.mid.attn_1 = IrisAttnBlock(block_in)
        self.mid.block_2 = IrisResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=timestep_embedding_channels,
            dropout=config.dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(config.num_res_blocks + 1):
                block.append(
                    IrisResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        timestep_embedding_channels=timestep_embedding_channels,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(IrisAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = IrisUpsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, config.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(
        self, z: torch.Tensor, output_hidden_states: bool = False, output_attentions: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        timestep_embedding = None  # timestep embedding

        # z to block_in
        h = self.conv_in(z)
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        hidden_states = hidden_states + (h,) if output_hidden_states else None
        # middle
        h = self.mid.block_1(h, timestep_embedding)
        # Add the hidden state to the tuple
        hidden_states = hidden_states + (h,) if output_hidden_states else None
        attn_out = self.mid.attn_1(h)
        h = attn_out[0]
        # Add attention weights
        attentions = attentions + (attn_out[1],) if output_attentions else None

        # Add the hidden state to the tuple
        hidden_states = hidden_states + (h,) if output_hidden_states else None
        h = self.mid.block_2(h, timestep_embedding)
        # Add the hidden state to the tuple
        hidden_states = hidden_states + (h,) if output_hidden_states else None

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, timestep_embedding)
                # Add the hidden state to the tuple
                hidden_states = hidden_states + (h,) if output_hidden_states else None
                if len(self.up[i_level].attn) > 0:
                    attn_out = self.up[i_level].attn[i_block](h)
                    h = attn_out[0]
                    # Add attention weights
                    attentions = attentions + (attn_out[1],) if output_attentions else None
                    # Add the hidden state to the tuple
                    hidden_states = hidden_states + (h,) if output_hidden_states else None
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                # Add the hidden state to the tuple
                hidden_states = hidden_states + (h,) if output_hidden_states else None

        # end
        h = self.norm_out(h)
        h = ACT2FN["swish"](h)
        h = self.conv_out(h)

        return h, hidden_states, attentions


class IrisUpsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class IrisDownsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class IrisResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float,
        timestep_embedding_channels: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if timestep_embedding_channels > 0:
            self.temb_proj = nn.Linear(timestep_embedding_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = ACT2FN["swish"](h)
        h = self.conv1(h)

        if timestep_embedding is not None:
            h = h + self.temb_proj(ACT2FN["swish"](timestep_embedding))[:, :, None, None]

        h = self.norm2(h)
        h = ACT2FN["swish"](h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class IrisAttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # q:query, k:key, v:value
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        batch_size, num_channels, height, width = q.shape
        q = q.reshape(batch_size, num_channels, height * width)
        q = q.permute(0, 2, 1)  # batch_size,height * width,num_channels
        k = k.reshape(batch_size, num_channels, height * width)  # batch_size,num_channels,height * width
        w_ = torch.bmm(
            q, k
        )  # batch_size,height * width,height * width    width[batch_size,i,j]=sum_c q[batch_size,i,num_channels]k[batch_size,num_channels,j]
        w_ = w_ * (int(num_channels) ** (-0.5))
        # attention weights
        w_ = nn.functional.softmax(w_, dim=2)
        attention_weights = w_

        # attend to values
        v = v.reshape(batch_size, num_channels, height * width)
        w_ = w_.permute(0, 2, 1)  # batch_size,height * width,height * width (first height * width of k, second of q)
        h_ = torch.bmm(
            v, w_
        )  # batch_size, num_channels,height * width (height * width of q) h_[batch_size,num_channels,j] = sum_i v[batch_size,num_channels,i] w_[batch_size,i,j]
        h_ = h_.reshape(batch_size, num_channels, height, width)

        h_ = self.proj_out(h_)

        return (x + h_,) + (attention_weights,)


class IrisLPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = IrisScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        self.net = IrisVgg16(requires_grad=False)
        self.lin0 = IrisNetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = IrisNetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = IrisNetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = IrisNetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = IrisNetLinLayer(self.chns[4], use_dropout=use_dropout)
        for param in self.parameters():
            param.requires_grad = False

    def normalize_tensor(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def spatial_average(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        return x.mean([2, 3], keepdim=keepdim)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = self.normalize_tensor(outs0[kk]), self.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [self.spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]
        return val


class IrisScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class IrisNetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [nn.Dropout()] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class IrisVgg16(nn.Module):
    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class IrisDiscreteAutoEncoder(nn.Module):
    """A discrete autoencoder based on the implementation of VQGAN but without its discriminator(not to be confused with transformers discrete_autoencoder)"""

    def __init__(
        self, vocab_size: int, embed_dim: int, encoder: IrisEncoder, decoder: IrisDecoder, with_lpips: bool = True
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = IrisLPIPS().eval() if with_lpips else None

    def forward(
        self,
        x: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ) -> Tuple[torch.Tensor]:
        outputs, all_hidden_states_enc, attentions_enc = self.encode(
            x, should_preprocess, output_hidden_states, output_attentions
        )
        decoder_input = outputs[0] + (outputs[1] - outputs[0]).detach()
        reconstructions, all_hidden_states_dec, attentions_dec = self.decode(
            decoder_input, should_postprocess, output_hidden_states, output_attentions
        )
        return (
            (outputs[0], outputs[1], reconstructions),
            all_hidden_states_enc + all_hidden_states_dec if output_hidden_states else None,
            attentions_enc + attentions_dec if output_attentions else None,
        )

    def encode(
        self,
        x: torch.Tensor,
        should_preprocess: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., num_channels, height, width)
        x = x.view(-1, *shape[-3:])
        z, all_hidden_states, attentions = self.encoder(x, output_hidden_states, output_attentions)
        z = self.pre_quant_conv(z)
        batch_size, num_tokens_per_frame, height, width = z.shape
        z_flattened = torch.flatten(z.permute(0, 2, 3, 1), end_dim=2).contiguous()
        dist_to_embeddings = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = (
            self.embedding(tokens)
            .view(batch_size, height, width, num_tokens_per_frame)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return (z, z_q, tokens), all_hidden_states, attentions

    def decode(
        self,
        z_q: torch.Tensor,
        should_postprocess: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        shape = z_q.shape  # (..., num_tokens_per_frame, height, width)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec, all_hidden_states, attentions = self.decoder(z_q, output_hidden_states, output_attentions)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec, all_hidden_states, attentions

    @torch.no_grad()
    def encode_decode(
        self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False
    ) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess)[0][1]
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)


class IrisKeysValues:
    def __init__(
        self, num_samples: int, max_tokens: int, num_layers: int, device: torch.device, config: object
    ) -> None:
        self._keys_values = StaticCache(
            config=config, max_batch_size=num_samples, max_cache_len=max_tokens, device=device
        )
        self._size = []
        self.num_layers = num_layers
        self.device = device
        self.reset()

    def __getitem__(self, key: int) -> StaticCache:
        return self._keys_values.key_cache[key][:, :, : self._size[key], :], self._keys_values.value_cache[key][
            :, :, : self._size[key], :
        ]

    def __len__(self):
        return len(self._keys_values.key_cache)

    @property
    def size(self):
        return (
            self._keys_values.max_batch_size,
            self._keys_values.num_key_value_heads,
            self._size,
            self._keys_values.head_dim,
        )

    def reset(self) -> None:
        self._keys_values.reset()
        self._size = [0] * self.num_layers

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_num: int) -> None:
        self._keys_values.key_cache[layer_num], self._keys_values.value_cache[layer_num] = self._keys_values.update(
            k, v, layer_num, cache_kwargs={"cache_position": torch.arange(k.shape[2], device=self.device)}
        )
        if self._size[layer_num] != k.size(2):
            self._size[layer_num] += k.size(2)


class IrisTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([IrisBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim_world_model)

    def generate_empty_keys_values(self, num_samples: int, max_tokens: int, config: object) -> IrisKeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return IrisKeysValues(num_samples, max_tokens, self.config.num_layers, device, config)

    def forward(
        self,
        sequences: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        past_keys_values: Optional[IrisKeysValues] = None,
    ) -> torch.Tensor:
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        # pos and token embeddings to return in hidden states
        x = self.drop(sequences)
        hidden_states = hidden_states + (x,) if output_hidden_states else None
        for i, block in enumerate(self.blocks):
            x, attention = block(x, None if past_keys_values is None else past_keys_values, i)
            # Add attention weights (shape: (batch_size, num_heads, target_sequence_length, source_sequence_length))
            attentions = attentions + (attention,) if output_attentions else None
            # Add hidden state to the tuple
            hidden_states = hidden_states + (x,) if output_hidden_states else None

        x = self.ln_f(x)
        hidden_states = hidden_states + (x,) if output_hidden_states else None

        return x, hidden_states, attentions


class IrisBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim_world_model)
        self.ln2 = nn.LayerNorm(config.embed_dim_world_model)
        self.attn = IrisSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim_world_model, 4 * config.embed_dim_world_model),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim_world_model, config.embed_dim_world_model),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self, x: torch.Tensor, past_keys_values: Optional[IrisKeysValues] = None, layer_num: int = None
    ) -> torch.Tensor:
        x_attn, attentions = self.attn(self.ln1(x), past_keys_values, layer_num)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x, attentions


class IrisSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.embed_dim_world_model % config.num_heads == 0
        assert config.attention in ("causal", "block_causal")
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model)
        self.query = nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model)
        self.value = nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(
            causal_mask,
            torch.block_diag(
                *[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]
            ),
        )
        self.register_buffer("mask", causal_mask if config.attention == "causal" else block_causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[StaticCache] = None, layer_num: int = None) -> torch.Tensor:
        batch_size, num_timesteps, num_z_channels = x.size()

        if kv_cache is not None:
            num_samples, num_heads, max_tokens, embed_dim = kv_cache.size
            assert (
                num_heads == self.num_heads and num_samples == batch_size and embed_dim * num_heads == num_z_channels
            )
        else:
            max_tokens = 0
        max_tokens = max_tokens[layer_num] if kv_cache is not None else max_tokens

        query = (
            self.query(x)
            .view(batch_size, num_timesteps, self.num_heads, num_z_channels // self.num_heads)
            .transpose(1, 2)
        )  # (batch_size, num_heads, num_timesteps, embed_dim)
        key = (
            self.key(x)
            .view(batch_size, num_timesteps, self.num_heads, num_z_channels // self.num_heads)
            .transpose(1, 2)
        )  # (batch_size, num_heads, num_timesteps, embed_dim)
        value = (
            self.value(x)
            .view(batch_size, num_timesteps, self.num_heads, num_z_channels // self.num_heads)
            .transpose(1, 2)
        )  # (batch_size, num_heads, num_timesteps, embed_dim)

        if kv_cache is not None:
            kv_cache.update(key, value, layer_num)
            key, value = kv_cache[layer_num]

        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att = att.masked_fill(
            self.mask[max_tokens : max_tokens + num_timesteps, : max_tokens + num_timesteps] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ value

        y = torch.flatten(y.permute(0, 2, 1, 3), start_dim=2).contiguous()

        y = self.resid_drop(self.proj(y))

        return (y, att)


class IrisWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = IrisTransformer(config)

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim_world_model)

        self.embedder = IrisEmbedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [
                    nn.Embedding(act_vocab_size, config.embed_dim_world_model),
                    nn.Embedding(obs_vocab_size, config.embed_dim_world_model),
                ]
            ),
        )

        self.head_observations = IrisHead(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model),
                nn.ReLU(),
                nn.Linear(config.embed_dim_world_model, obs_vocab_size),
            ),
        )

        self.head_rewards = IrisHead(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model),
                nn.ReLU(),
                nn.Linear(config.embed_dim_world_model, 3),
            ),
        )

        self.head_ends = IrisHead(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim_world_model, config.embed_dim_world_model),
                nn.ReLU(),
                nn.Linear(config.embed_dim_world_model, 2),
            ),
        )

    def forward(
        self,
        tokens: torch.LongTensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        past_keys_values: Optional[IrisKeysValues] = None,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size[2][0]

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device)
        )
        x, transformer_hidden_states, attentions = self.transformer(
            sequences, output_hidden_states, output_attentions, past_keys_values=past_keys_values
        )
        logits_observations, head_hidden_states_observations = self.head_observations(
            x, output_hidden_states, num_steps=num_steps, prev_steps=prev_steps
        )
        logits_rewards, head_hidden_states_rewards = self.head_rewards(
            x, output_hidden_states, num_steps=num_steps, prev_steps=prev_steps
        )
        logits_ends, head_hidden_states_ends = self.head_ends(
            x, output_hidden_states, num_steps=num_steps, prev_steps=prev_steps
        )

        return (
            (x, logits_observations, logits_rewards, logits_ends),
            transformer_hidden_states
            + head_hidden_states_observations
            + head_hidden_states_rewards
            + head_hidden_states_ends
            if output_hidden_states
            else None,
            attentions,
        )

    def compute_labels_world_model(
        self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = torch.flatten(
            obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), start_dim=1
        ).contiguous()[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)


class IrisWorldModelImagine:
    def __init__(
        self,
        discrete_autoencoder: nn.Module,
        world_model: nn.Module,
        device: Union[str, torch.device],
    ) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.discrete_autoencoder = discrete_autoencoder.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(
        self, obs_tokens: torch.LongTensor, config: object
    ) -> torch.FloatTensor:
        num_samples, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
            num_samples=num_samples, max_tokens=self.world_model.config.max_tokens, config=config
        )
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)[0]
        return outputs_wm[0]  # (B, K, E)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor, config) -> torch.FloatTensor:
        obs_tokens = self.discrete_autoencoder.encode(observations, should_preprocess=True)[0][
            2
        ]  # (batch_size, num_channels, height, width) -> (batch_size, num_tokens_per_frame)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens, config)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def step(
        self,
        action: Union[int, np.ndarray, torch.LongTensor],
        should_predict_next_obs: bool = True,
        config: object = None,
    ) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None
        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size[2][0] + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens, config)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (batch_size, 1)

        for k in range(num_passes):  # assumption that there is only one action token.
            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)[0]
            output_sequence.append(outputs_wm[0])

            if k == 0:
                reward = (
                    Categorical(logits=outputs_wm[2]).sample().float().cpu().numpy().reshape(-1) - 1
                )  # (batch_size,)
                done = (
                    Categorical(logits=outputs_wm[3]).sample().cpu().numpy().astype(bool).reshape(-1)
                )  # (batch_size,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm[1]).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)  # (batch_size, 1 + num_tokens_per_frame, embed_dim)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)  # (batch_size, num_tokens_per_frame)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, None

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.discrete_autoencoder.embedding(
            self.obs_tokens
        )  # (batch_size, num_tokens_per_frame, embed_dim)
        h = int(np.sqrt(self.num_observations_tokens))
        z = embedded_tokens.permute(0, 2, 1)
        z = z.view(z.shape[0], z.shape[1], h, z.shape[2] // h).contiguous()

        rec = self.discrete_autoencoder.decode(z, should_postprocess=True)[
            0
        ]  # (batch_size, num_channels, height, width)
        return torch.clamp(rec, 0, 1)


class IrisActorCritic(nn.Module):
    def __init__(self, act_vocab_size, use_original_obs: bool = False) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_vocab_size)

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(
        self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None
    ) -> None:
        device = self.conv1.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert (
                burnin_observations.ndim == 5
                and burnin_observations.size(0) == n
                and mask_padding is not None
                and burnin_observations.shape[:2] == mask_padding.shape
            )
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding=mask_padding[:, i])[0]

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def compute_lambda_returns(self, rewards, values, ends, gamma, lambda_):
        assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
        assert (
            rewards.shape == ends.shape == values.shape
        ), f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
        t = rewards.size(1)
        lambda_returns = torch.empty_like(values)
        lambda_returns[:, -1] = values[:, -1]
        lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

        last = values[:, -1]
        for i in list(range(t - 1))[::-1]:
            lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
            last = lambda_returns[:, i]

        return lambda_returns

    def forward(
        self,
        inputs: torch.FloatTensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        mask_padding: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], None]:
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
        assert mask_padding is None or (
            mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any()
        )
        x = inputs[mask_padding] if mask_padding is not None else inputs

        x = x.mul(2).sub(1)
        hidden_states = () if output_hidden_states else None
        x = F.relu(self.maxp1(self.conv1(x)))
        hidden_states = hidden_states + (x,) if output_hidden_states else None
        x = F.relu(self.maxp2(self.conv2(x)))
        hidden_states = hidden_states + (x,) if output_hidden_states else None
        x = F.relu(self.maxp3(self.conv3(x)))
        hidden_states = hidden_states + (x,) if output_hidden_states else None
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        hidden_states = hidden_states + (x,) if output_hidden_states else None
        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        hidden_states = hidden_states + (self.hx,) if output_hidden_states else None

        logits_actions = self.actor_linear(self.hx).unsqueeze(1).contiguous()
        means_values = self.critic_linear(self.hx).unsqueeze(1).contiguous()

        return (logits_actions, means_values), hidden_states, None

    def imagine(
        self,
        batch: Batch,
        discrete_autoencoder: IrisDiscreteAutoEncoder,
        world_model: IrisWorldModel,
        horizon: int,
        show_pbar: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        config: object = None,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        assert not self.use_original_obs
        initial_observations = batch["observations"]
        mask_padding = batch["mask_padding"]
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_imagine = IrisWorldModelImagine(discrete_autoencoder, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        burnin_observations = (
            torch.clamp(
                discrete_autoencoder.encode_decode(
                    initial_observations[:, :-1], should_preprocess=True, should_postprocess=True
                )[0],
                0,
                1,
            )
            if initial_observations.size(1) > 1
            else None
        )

        self.reset(
            n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1]
        )

        obs = wm_imagine.reset_from_initial_observations(initial_observations[:, -1], config)
        for k in tqdm(range(horizon), disable=not show_pbar, desc="Imagination", file=sys.stdout):
            all_observations.append(obs)

            outputs_ac, hidden_states, _ = self(obs, output_hidden_states, output_attentions)
            action_token = Categorical(logits=outputs_ac[0]).sample()
            obs, reward, done, _ = wm_imagine.step(
                action_token, should_predict_next_obs=(k < horizon - 1), config=config
            )

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac[0])
            all_values.append(outputs_ac[1])
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return (
            (
                torch.stack(all_observations, dim=1)
                .mul(255)
                .byte(),  # (batch_size, num_timesteps, num_channels, height, width) in [0, 255]
                torch.cat(all_actions, dim=1),  # (batch_size, num_timesteps)
                torch.cat(all_logits_actions, dim=1),  # (batch_size, num_timesteps, #actions)
                torch.cat(all_values, dim=1).squeeze(2).contiguous(),  # (batch_size, num_timesteps)
                torch.cat(all_rewards, dim=1).to(device),  # (batch_size, num_timesteps)
                torch.cat(all_ends, dim=1).to(device),  # (batch_size, num_timesteps)
            ),
            outputs_ac,
            hidden_states,
        )


class IrisComponentLosses:
    """
    Loss functions for the Iris' components.
    """

    def compute_discrete_autoencoder_loss(
        self,
        discrete_autoencoder: IrisDiscreteAutoEncoder,
        batch: Batch,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> IrisLossWithIntermediateLosses:
        assert discrete_autoencoder.lpips is not None

        observations = discrete_autoencoder.preprocess_input(
            torch.flatten(batch["observations"], end_dim=1).contiguous()
        )
        outputs, all_hidden_states, all_attentions = discrete_autoencoder(
            observations, output_hidden_states, output_attentions, should_preprocess=False, should_postprocess=False
        )

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (outputs[0].detach() - outputs[1]).pow(2).mean() + beta * (
            outputs[0] - outputs[1].detach()
        ).pow(2).mean()

        reconstruction_loss = torch.abs(observations - outputs[2]).mean()
        perceptual_loss = torch.mean(discrete_autoencoder.lpips(observations, outputs[2]))

        return (
            IrisLossWithIntermediateLosses(
                commitment_loss=commitment_loss,
                reconstruction_loss=reconstruction_loss,
                perceptual_loss=perceptual_loss,
            ),
            outputs,
            all_hidden_states,
            all_attentions,
        )

    def compute_world_model_loss(
        self,
        world_model: IrisWorldModel,
        batch: Batch,
        discrete_autoencoder: IrisDiscreteAutoEncoder,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> IrisLossWithIntermediateLosses:
        with torch.no_grad():
            obs_tokens = discrete_autoencoder.encode(batch["observations"], should_preprocess=True)[0][2]  # (B, L, K)

        act_tokens = batch["actions"].unsqueeze(-1).contiguous()

        tokens = torch.flatten(torch.cat((obs_tokens, act_tokens), dim=2), start_dim=1).contiguous()  # (B, L(K+1))

        outputs, all_hidden_states, all_attentions = world_model(tokens, output_hidden_states, output_attentions)

        labels_observations, labels_rewards, labels_ends = world_model.compute_labels_world_model(
            obs_tokens, batch["rewards"], batch["ends"], batch["mask_padding"]
        )

        logits_observations = torch.flatten(outputs[1][:, :-1], end_dim=1).contiguous()
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy((torch.flatten(outputs[2], end_dim=1).contiguous()), labels_rewards)
        loss_ends = F.cross_entropy((torch.flatten(outputs[3], end_dim=1).contiguous()), labels_ends)

        return (
            IrisLossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends),
            outputs,
            all_hidden_states,
            all_attentions,
        )

    def compute_actor_critic_loss(
        self,
        actor_critic: IrisActorCritic,
        config: IrisConfig,
        batch: Batch,
        discrete_autoencoder: IrisDiscreteAutoEncoder,
        world_model: IrisWorldModel,
        imagine_horizon: int,
        gamma: float,
        lambda_: float,
        entropy_weight: float,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> IrisLossWithIntermediateLosses:
        assert not actor_critic.use_original_obs
        outputs, outputs_ac, hidden_states = actor_critic.imagine(
            batch,
            discrete_autoencoder,
            world_model,
            horizon=imagine_horizon,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            config=config,
        )

        with torch.no_grad():
            lambda_returns = actor_critic.compute_lambda_returns(
                rewards=outputs[4],
                values=outputs[3],
                ends=outputs[5],
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs[3][:, :-1]

        d = Categorical(logits=outputs[2][:, :-1])
        log_probs = d.log_prob(outputs[1][:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = -entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return (
            IrisLossWithIntermediateLosses(
                loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy
            ),
            outputs_ac,
            hidden_states,
            None,
        )


class IrisRlAgent(nn.Module):
    def __init__(
        self, discrete_autoencoder: IrisDiscreteAutoEncoder, world_model: IrisWorldModel, actor_critic: IrisActorCritic
    ):
        super().__init__()
        self.discrete_autoencoder = discrete_autoencoder
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.device = self.actor_critic.conv1.weight.device

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = (
            obs
            if self.actor_critic.use_original_obs
            else torch.clamp(
                self.discrete_autoencoder.encode_decode(obs, should_preprocess=True, should_postprocess=True)[0], 0, 1
            )
        )
        logits_actions = self.actor_critic(input_ac)[0][0][:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token


@add_start_docstrings("The IRIS Model", IRIS_START_DOCSTRING)
class IrisModel(IrisPreTrainedModel):
    """

    The model,IRIS (Imagination with auto-Regression over an Inner Speech), a data-efficient rl_agent that learns in a world model composed of a
    discrete autoencoder and an autoregressive Transformer.It learns behaviors by accurately simulating millions of trajectories
    Refer to the paper for more details: https://arxiv.org/abs/2209.00588

    """

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.model_device = config.device
        encoder = IrisEncoder(config)
        decoder = IrisDecoder(config)
        discrete_autoencoder = IrisDiscreteAutoEncoder(
            self.config.vocab_size,
            self.config.embed_dim_discrete_autoencoder,
            encoder,
            decoder,
        )
        world_model = IrisWorldModel(
            obs_vocab_size=discrete_autoencoder.vocab_size,
            act_vocab_size=self.config.num_actions,
            config=config,
        )
        actor_critic = IrisActorCritic(
            act_vocab_size=self.config.num_actions,
            use_original_obs=self.config.use_original_obs_actor_critic,
        )
        self.rl_agent = IrisRlAgent(discrete_autoencoder, world_model, actor_critic)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(IRIS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=IrisOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        observations: torch.FloatTensor = None,
        actions: torch.FloatTensor = None,
        rewards: torch.FloatTensor = None,
        ends: torch.IntTensor = None,
        mask_padding: torch.BoolTensor = None,
        component: str = None,
        should_preprocess: Optional[bool] = None,
        should_postprocess: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], IrisOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import IrisModel
        >>> import torch
        >>> import gym

        >>> model = IrisModel.from_pretrained("ruffy369/iris-breakout")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> batch_size = 1 #here, single batch size is for the sake of example asin original code all the three components have separate batch size
        >>> sequence_length_discrete_autoencoder = 1
        >>> sequence_length_world_model = 20
        >>> sequence_length_actor_critic = 21
        >>> env = gym.make("BreakoutNoFrameskip-v4")
        >>> observations_tok, actions_tok, rewards_tok, dones_tok = [], [], [], []
        >>> observations_wm, actions_wm, rewards_wm, dones_wm = [], [], [], []
        >>> observations_ac, actions_ac, rewards_ac, dones_ac = [], [], [], []
        >>> observation_tok_batch, actions_tok_batch, rewards_tok_batch, dones_tok_batch = [], [], [], []
        >>> observation_wm_batch, actions_wm_batch, rewards_wm_batch, dones_wm_batch = [], [], [], []
        >>> observation_ac_batch, actions_ac_batch, rewards_ac_batch, dones_ac_batch = [], [], [], []


        >>> for b in batch_size:
        ...     for t in sequence_length_discrete_autoencoder:
        ...         act = torch.randint(0,4,(batch_size,1)).cpu().numpy() #use rl_agent.act with obs from env.reset in training
        ...         obs, reward, done, _ = env.step(act)
        ...         obs = torch.FloatTensor(obs).div(255).permute(0,3,1,2).cpu().numpy()
        ...         actions_tok.append(act.tolist())
        ...         rewards_tok.append(reward)
        ...         dones_tok.append(done)
        ...         observations_tok.append(obs.tolist())
        ...     observation_tok_batch.append(observations_tok)
        ...     actions_tok_batch.apped(actions_tok)
        ...     rewards_tok_batch.append(rewards_tok)
        ...     dones_tok_batch.append(dones_tok)

        ...     for t in sequence_length_world_model:
        ...         act = torch.randint(0,4,(batch_size,1)).cpu().numpy() #use rl_agent.act with obs from env.reset in training
        ...         obs, reward, done, _ = env.step(act)
        ...         obs = torch.FloatTensor(obs).div(255).permute(0,3,1,2).cpu().numpy()
        ...         actions_wm.append(act.tolist())
        ...         rewards_wm.append(reward)
        ...         dones_wm.append(done)
        ...         observations_wm.append(obs.tolist())
        ...     observation_wm_batch.append(observations_wm)
        ...     actions_wm_batch.append(actions_wm)
        ...     rewards_wm_batch.append(rewards_wm)
        ...     dones_wm_batch.append(dones_wm)

        ...     for t in sequence_length_actor_critic:
        ...         act = torch.randint(0,4,(batch_size,1)).cpu().numpy() #use rl_agent.act with obs from env.reset in training
        ...         obs, reward, done, _ = env.step(act)
        ...         obs = torch.FloatTensor(obs).div(255).permute(0,3,1,2).cpu().numpy()
        ...         actions_ac.append(act.tolist())
        ...         rewards_ac.append(reward)
        ...         dones_ac.append(done)
        ...         observations_ac.append(obs.tolist())
        ...     observation_ac_batch.append(observations_ac)
        ...     actions_ac_batch.append(dones_ac)
        ...     rewards_ac_batch.append(rewards_ac)
        ...     dones_ac_batch.append(actions_ac)

        >>> observation_tok_batch = torch.tensor(observation_tok_batch).float().to(device)
        >>> actions_tok_batch = torch.tensor(actions_tok_batch).long().to(device)
        >>> rewards_tok_batch = torch.tensor(rewards_tok_batch).float().to(device)
        >>> dones_tok_batch = torch.tensor(dones_tok_batch).long().to(device)

        >>> observation_wm_batch = torch.tensor(observation_wm_batch).float().to(device)
        >>> actions_wm_batch = torch.tensor(actions_wm_batch).long().to(device)
        >>> rewards_wm_batch = torch.tensor(rewards_wm_batch).float().to(device)
        >>> dones_wm_batch = torch.tensor(dones_wm_batch).long().to(device)

        >>> observation_ac_batch = torch.tensor(observation_ac_batch).float().to(device)
        >>> actions_ac_batch = torch.tensor(actions_ac_batch).long().to(device)
        >>> rewards_ac_batch = torch.tensor(rewards_ac_batch).float().to(device)
        >>> dones_ac_batch = torch.tensor(dones_ac_batch).long().to(device)

        >>> mask_padding_tok = torch.ones(batch_size,sequence_length_discrete_autoencoder).bool().to(device)
        >>> mask_padding_wm = torch.ones(batch_size,sequence_length_world_model).bool().to(device)
        >>> mask_padding_ac = torch.ones(batch_size,sequence_length_actor_critic).bool().to(device)

        >>> observations = [observation_tok_batch,observation_wm_batch,observation_ac_batch]
        >>> actions = [actions_tok_batch,actions_wm_batch,actions_ac_batch]
        >>> rewards = [rewards_tok_batch,rewards_wm_batch,rewards_ac_batch]
        >>> ends = [dones_tok_batch,dones_wm_batch,dones_ac_batch]
        >>> mask_padding = [mask_padding_tok,mask_padding_wm,mask_padding_ac]


        >>> # forward pass
        >>> with torch.no_grad():
        ...     model_pred = model(
        ...         observations = observations,
                    actions = actions,
                    rewards = rewards,
                    ends = ends,
                    mask_padding = mask_padding,
                    should_preprocess = True,
                    should_postprocess = True,
                    output_hidden_states = False,
                    output_attentions = False,
        ...     )

        >>> # model_pred is the output instance of IrisOutput() class. These are the attributes if output is returned as dict and in same order if returned as tuple:
        >>> # reconstructed_img, losses, action_preds, reward_preds, epsiode_end, obs_preds, hidden_states, attentions
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if component is not None:
            assert component in ("discrete_autoencoder", "world_model", "actor_critic")

        cfg_discrete_autoencoder = {
            "should_preprocess": should_preprocess,
            "should_postprocess": should_postprocess,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
        }
        cfg_world_model = {
            "discrete_autoencoder": self.rl_agent.discrete_autoencoder,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
        }
        cfg_actor_critic = {
            "discrete_autoencoder": self.rl_agent.discrete_autoencoder,
            "world_model": self.rl_agent.world_model,
            "imagine_horizon": self.config.imagine_horizon_train_actor_critic,
            "gamma": self.config.gamma,
            "lambda_": self.config.lambda_,
            "entropy_weight": self.config.entropy_weight,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
        }

        if component is None and isinstance(observations, List):
            batch_discrete_autoencoder = {
                "observations": observations[0],
                "actions": actions[0],
                "rewards": rewards[0],
                "ends": ends[0],
                "mask_padding": mask_padding[0],
            }
            batch_world_model = {
                "observations": observations[1],
                "actions": actions[1],
                "rewards": rewards[1],
                "ends": ends[1],
                "mask_padding": mask_padding[1],
            }
            batch_actor_critic = {
                "observations": observations[2],
                "actions": actions[2],
                "rewards": rewards[2],
                "ends": ends[2],
                "mask_padding": mask_padding[2],
            }
        else:
            batch = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "ends": ends,
                "mask_padding": mask_padding,
            }

        all_hidden_states_discrete_autoencoder, all_hidden_states_world_model, all_hidden_states_actor_critic = (
            (),
            (),
            (),
        )
        all_attentions_discrete_autoencoder, all_attentions_world_model = (
            (),
            (),
        )

        component_losses = IrisComponentLosses()

        if component is None and isinstance(observations, List):
            (
                losses_discrete_autoencoder,
                discrete_autoencoder_outputs,
                all_hidden_states_discrete_autoencoder,
                all_attentions_discrete_autoencoder,
            ) = component_losses.compute_discrete_autoencoder_loss(
                self.rl_agent.discrete_autoencoder, batch_discrete_autoencoder, **cfg_discrete_autoencoder
            )
            losses_discrete_autoencoder = losses_discrete_autoencoder / self.config.grad_acc_steps_discrete_autoencoder

            (
                losses_world_model,
                world_model_outputs,
                all_hidden_states_world_model,
                all_attentions_world_model,
            ) = component_losses.compute_world_model_loss(
                self.rl_agent.world_model, batch_world_model, **cfg_world_model
            )
            losses_world_model = losses_world_model / self.config.grad_acc_steps_world_model

            (
                losses_actor_critic,
                actor_critic_outputs,
                all_hidden_states_actor_critic,
                _,
            ) = component_losses.compute_actor_critic_loss(
                self.rl_agent.actor_critic, self.config, batch_actor_critic, **cfg_actor_critic
            )
            losses_actor_critic = losses_actor_critic / self.config.grad_acc_steps_actor_critic

            losses = (losses_discrete_autoencoder, losses_world_model, losses_actor_critic)
            reconstructed_img = discrete_autoencoder_outputs[2]
            action_preds = actor_critic_outputs[0]
            reward_preds = world_model_outputs[2]
            epsiode_end = world_model_outputs[3]
            obs_preds = world_model_outputs[1]
        else:
            if component == "discrete_autoencoder":
                (
                    losses,
                    discrete_autoencoder_outputs,
                    all_hidden_states_discrete_autoencoder,
                    all_attentions_discrete_autoencoder,
                ) = component_losses.compute_discrete_autoencoder_loss(
                    self.rl_agent.discrete_autoencoder, batch, **cfg_discrete_autoencoder
                )
                losses = losses / self.config.grad_acc_steps_discrete_autoencoder
            elif component == "world_model":
                (
                    losses,
                    world_model_outputs,
                    all_hidden_states_world_model,
                    all_attentions_world_model,
                ) = component_losses.compute_world_model_loss(self.rl_agent.world_model, batch, **cfg_world_model)
                losses = losses / self.config.grad_acc_steps_world_model
            else:
                (
                    losses,
                    actor_critic_outputs,
                    all_hidden_states_actor_critic,
                    _,
                ) = component_losses.compute_actor_critic_loss(
                    self.rl_agent.actor_critic, self.config, batch, **cfg_actor_critic
                )
                losses = losses / self.config.grad_acc_steps_actor_critic

            reconstructed_img = discrete_autoencoder_outputs[2] if component == "discrete_autoencoder" else None
            action_preds = actor_critic_outputs[0] if component == "actor_critic" else None
            reward_preds = world_model_outputs[2] if component == "world_model" else None
            epsiode_end = world_model_outputs[3] if component == "world_model" else None
            obs_preds = world_model_outputs[1] if component == "world_model" else None

        all_hidden_states = (
            all_hidden_states_discrete_autoencoder + all_hidden_states_world_model + all_hidden_states_actor_critic
            if output_hidden_states
            else None
        )
        all_self_attentions = (
            all_attentions_discrete_autoencoder + all_attentions_world_model if output_attentions else None
        )

        if output_hidden_states:
            for hidden_state in all_hidden_states:
                hidden_state.requires_grad_(True)

        if output_attentions:
            for attention in all_self_attentions:
                attention.requires_grad_(True)

        if not return_dict:
            return tuple(
                v
                for v in [
                    reconstructed_img,
                    losses,
                    action_preds,
                    reward_preds,
                    epsiode_end,
                    obs_preds,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return IrisOutput(
            reconstructed_img=reconstructed_img,
            losses=losses,
            action_preds=action_preds,
            reward_preds=reward_preds,
            epsiode_end=epsiode_end,
            obs_preds=obs_preds,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
