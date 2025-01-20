# coding=utf-8
# Copyright 2025  Apple and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch AIMv2 model."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings
from .configuration_aimv2 import AIMv2Config


__all__ = ["AIMv2Model", "AIMv2PreTrainedModel"]

_CONFIG_FOR_DOC = "AIMv2Config"

AIMV2_START_DOCSTRING = r"""

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`AIMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

AIMV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`~AutoImageProcessor.__call__`] for details.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to apply to the attention scores. A value of 1 indicates the position is not masked, and a value of 0
            indicates the position is masked.

            <Tip>

            What is the mask? Most models expect a value of 1, indicating the position *should* attend, and 0,
            indicating the position *should not* attend. For example, if your input sequence length is 5 and you only
            want to attend to the first 3 positions, the mask should be `[1, 1, 1, 0, 0]`.

            </Tip>

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class AIMv2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self._norm(hidden_states.float()).type_as(hidden_states)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.silu(self.fc1(hidden_states)) * self.fc3(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AIMv2PatchEmbed(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = AIMv2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AIMv2ViTPreprocessor(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = AIMv2PatchEmbed(config)
        self.pos_embed = nn.Parameter(torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(pixel_values)
        _, num_tokens, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :num_tokens]
        return tokens


class AIMv2Attention(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv = nn.Linear(hidden_size, self.all_head_size * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(self.all_head_size, hidden_size, bias=config.use_bias)
        self.proj_drop = nn.Dropout(config.projection_dropout)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, seq_length, 3, self.num_attention_heads, self.attention_head_size)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)

        context_layer = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)

        return context_layer


class AIMv2Block(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.attn = AIMv2Attention(config)
        self.norm_1 = AIMv2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config)
        self.norm_2 = AIMv2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm_1(hidden_states), mask)
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states


class AIMv2Transformer(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.blocks = nn.ModuleList([AIMv2Block(config) for _ in range(config.num_hidden_layers)])
        self.post_trunk_norm = AIMv2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                hidden_states += (tokens,)

            tokens = block(tokens, mask)

        if output_hidden_states:
            hidden_states += (tokens,)

        tokens = self.post_trunk_norm(tokens)
        return tokens, hidden_states


@add_start_docstrings(
    "The bare AIMv2 Model transformer outputting raw hidden-states without any specific head on top.",
    AIMV2_START_DOCSTRING,
)
class AIMv2PreTrainedModel(PreTrainedModel):
    config_class = AIMv2Config
    base_model_prefix = "aimv2"
    main_input_name = "pixel_values"
    _no_split_modules = ["AIMv2ViTPreprocessor", "AIMv2Block"]
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            # Use Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AIMv2ViTPreprocessor):
            module.pos_embed.data = nn.init.trunc_normal_(
                module.pos_embed.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.pos_embed.dtype)


@add_start_docstrings(
    "The bare AIMv2 Model transformer outputting raw hidden-states without any specific head on top.",
    AIMV2_START_DOCSTRING,
)
class AIMv2Model(AIMv2PreTrainedModel):
    def __init__(self, config: AIMv2Config):
        super().__init__(config)
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AIMv2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, AIMv2ViTPreprocessor):
            module.pos_embed.data = nn.init.trunc_normal_(
                module.pos_embed.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.pos_embed.dtype)

    def get_input_embeddings(self) -> AIMv2PatchEmbed:
        return self.preprocessor.patchifier

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:
        """
             Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
                [`~AutoImageProcessor.__call__`] for details.

            mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to apply to the attention scores. A value of 1 indicates the position is not masked, and a value of 0
                indicates the position is masked.

                <Tip>

                What is the mask? Most models expect a value of 1, indicating the position *should* attend, and 0,
                indicating the position *should not* attend. For example, if your input sequence length is 5 and you only
                want to attend to the first 3 positions, the mask should be `[1, 1, 1, 0, 0]`.

                </Tip>

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*):
                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                 tensors for more detail.

        Returns:
            Returns a tuple if not dictionary if config.use_return_dict is set to True, else a tuple.
            x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Hidden states of the output at the output of the last layer of the model.
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True` is passed or if `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.

            Returns:

            Examples:
            ```python
            >>> from transformers import AutoImageProcessor, AIMv2Model
            >>> from PIL import Image
            >>> import requests

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-224")
            >>> model = AIMv2Model.from_pretrained("apple/aimv2-large-patch14-224")

            >>> inputs = processor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.preprocessor(pixel_values)
        x, hidden_states = self.trunk(x, mask, output_hidden_states=output_hidden_states)

        if not return_dict:
            res = (x,)
            res += (hidden_states,) if output_hidden_states else ()
            return res

        return BaseModelOutputWithNoAttention(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
        )
