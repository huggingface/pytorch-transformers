# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch TimeSformer model."""

import collections
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TimesformerConfig"
_CHECKPOINT_FOR_DOC = "facebook/timesformer"


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L155
class TimesformerPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)

        embeddings = self.projection(pixel_values)
        patch_width = embeddings.size(-1)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings, num_frames, patch_width


class TimesformerEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type

        self.attention_type = attention_type
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if attention_type != "space_only":
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]

        # create patch embeddings
        embeddings, num_frames, patch_width = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if embeddings.size(1) != self.position_embeddings.size(1):
            position_embeddings = self.position_embeddings
            cls_pos_embed = position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = position_embeddings[0, 1:, :].unsqueeze(0).transpose(1, 2)
            patch_num = int(other_pos_embed.size(2) ** 0.5)
            patch_height = embeddings.size(1) // patch_width
            other_pos_embed = other_pos_embed.reshape(1, embeddings.size(2), patch_num, patch_num)
            new_pos_embed = nn.functional.interpolate(
                other_pos_embed, size=(patch_height, patch_width), mode="nearest"
            )
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            embeddings = embeddings + new_pos_embed
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.pos_drop(embeddings)

        # Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = embeddings[:batch_size, 0, :].unsqueeze(1)
            embeddings = embeddings[:, 1:]
            _, patch_height, patch_width = embeddings.shape
            embeddings = (
                embeddings.reshape(batch_size, num_frames, patch_height, patch_width)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * patch_height, num_frames, patch_width)
            )
            # Resizing time embeddings in case they don't match
            if num_frames != self.time_embeddings.size(1):
                time_embeddings = self.time_embeddings.transpose(1, 2)
                new_time_embeddings = nn.functional.interpolate(time_embeddings, size=(num_frames), mode="nearest")
                new_time_embeddings = new_time_embeddings.transpose(1, 2)
                embeddings = embeddings + new_time_embeddings
            else:
                embeddings = embeddings + self.time_embeddings
            embeddings = self.time_drop(embeddings)
            embeddings = embeddings.view(batch_size, patch_height, num_frames, patch_width).reshape(
                batch_size, patch_height * num_frames, patch_width
            )
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings


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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->TimeSformer
class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L57
class TimesformerSelfAttention(nn.Module):
    def __init__(self, config: TimesformerConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)

    def forward(self, hidden_states, output_attentions: bool = False):
        batch_size, hidden_size, num_channels = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, hidden_size, num_channels)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimeSformerAttention(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.attention = TimesformerSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L39
class TimesformerIntermediate(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimesformerOutput(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L89
class TimesformerLayer(nn.Module):
    def __init__(self, config: TimesformerConfig, layer_index: int) -> None:
        super().__init__()

        attention_type = config.attention_type

        drop_path_rates = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        drop_path_rate = drop_path_rates[layer_index]

        self.drop_path = TimeSformerDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimesformerIntermediate(config)
        self.output = TimesformerOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.config = config
        self.attention_type = attention_type
        if attention_type not in ["divided_space_time", "space_only", "joint_space_time"]:
            raise ValueError("Unknown attention type: {}".format(attention_type))

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        num_frames = self.config.num_frames
        num_patch_width = self.config.image_size // self.config.patch_size
        batch_size = hidden_states.shape[0]
        num_spatial_tokens = (hidden_states.size(1) - 1) // num_frames
        num_patch_height = num_spatial_tokens // num_patch_width

        if self.attention_type in ["space_only", "joint_space_time"]:
            self_attention_outputs = self.attention(
                self.layernorm_before(hidden_states), output_attentions=output_attentions
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            hidden_states = hidden_states + self.drop_path(attention_output)

            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs

        elif self.attention_type == "divided_space_time":
            # Temporal
            temporal_embedding = hidden_states[:, 1:, :]
            temporal_embedding = temporal_embedding.reshape(
                batch_size, num_patch_height, num_patch_width, num_frames, temporal_embedding.shape[2]
            ).reshape(batch_size * num_patch_height * num_patch_width, num_frames, temporal_embedding.shape[2])

            temporal_attention_outputs = self.temporal_attention(
                self.temporal_layernorm(temporal_embedding),
            )
            attention_output = temporal_attention_outputs[0]

            residual_temporal = self.drop_path(attention_output)

            residual_temporal = residual_temporal.reshape(
                batch_size, num_patch_height, num_patch_width, num_frames, residual_temporal.shape[2]
            ).reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_temporal.shape[2])
            residual_temporal = self.temporal_dense(residual_temporal)
            temporal_embedding = hidden_states[:, 1:, :] + residual_temporal

            # Spatial
            init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, num_frames, 1)
            cls_token = cls_token.reshape(batch_size * num_frames, 1, cls_token.shape[2])
            spatial_embedding = temporal_embedding
            spatial_embedding = (
                spatial_embedding.reshape(
                    batch_size, num_patch_height, num_patch_width, num_frames, spatial_embedding.shape[2]
                )
                .permute(0, 3, 1, 2, 4)
                .reshape(batch_size * num_frames, num_patch_height * num_patch_width, spatial_embedding.shape[2])
            )
            spatial_embedding = torch.cat((cls_token, spatial_embedding), 1)

            spatial_attention_outputs = self.attention(
                self.layernorm_before(spatial_embedding), output_attentions=output_attentions
            )
            attention_output = spatial_attention_outputs[0]
            outputs = spatial_attention_outputs[1:]  # add self attentions if we output attention weights

            residual_spatial = self.drop_path(attention_output)

            # Taking care of CLS token
            cls_token = residual_spatial[:, 0, :]
            cls_token = cls_token.reshape(batch_size, num_frames, cls_token.shape[1])
            cls_token = torch.mean(cls_token, 1, True)  # averaging for every frame
            residual_spatial = residual_spatial[:, 1:, :]
            residual_spatial = (
                residual_spatial.reshape(
                    batch_size, num_frames, num_patch_height, num_patch_width, residual_spatial.shape[2]
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_spatial.shape[2])
            )
            residual = residual_spatial
            hidden_states = temporal_embedding

            # Mlp
            hidden_states = torch.cat((init_cls_token, hidden_states), 1) + torch.cat((cls_token, residual), 1)
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs


class TimesformerEncoder(nn.Module):
    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TimesformerLayer(config, ind) for ind in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)

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


class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TimesformerConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TimesformerLayer"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TimesformerEmbeddings):
            nn.init.trunc_normal_(module.cls_token, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)
            module.patch_embeddings.apply(self._init_weights)


TIMESFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TimesformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TIMESFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.preprocess`] for details.

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
    "The bare TimeSformer Model transformer outputting raw hidden-states without any specific head on top.",
    TIMESFORMER_START_DOCSTRING,
)
class TimesformerModel(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddings(config)
        self.encoder = TimesformerEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import av
        >>> import numpy as np

        >>> from transformers import AutoImageProcessor, TimesformerModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

        >>> # prepare video for the model
        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1569, 768]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """TimeSformer Model transformer with a video classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.""",
    TIMESFORMER_START_DOCSTRING,
)
class TimesformerForVideoClassification(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.timesformer = TimesformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoImageProcessor, TimesformerForVideoClassification
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        >>> model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     logits = outputs.logits

        >>> # model predicts one of the 400 Kinetics-400 classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        eating spaghetti
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.timesformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0][:, 0]

        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["TimesformerModel", "TimesformerForVideoClassification", "TimesformerPreTrainedModel"]
