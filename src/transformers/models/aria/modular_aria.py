# coding=utf-8
# Copyright 2024 The Rhymes-AI Teams Authors and The HuggingFace Inc. team. All rights reserved.
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
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...image_processing_utils import BaseImageProcessor, BatchFeature, select_best_resolution
from ...image_transforms import PaddingMode, convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils import (
    PreTokenizedInput,
    TextInput,
)
from ...utils import (
    TensorType,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_available
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ..llava.modeling_llava import LlavaCausalLMOutputWithPast
from ..llava_next.image_processing_llava_next import divide_to_patches, make_batched_images


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch import nn


def sequential_experts_gemm(token_states, expert_weights, tokens_per_expert):
    """
    Compute the matrix multiplication (GEMM) for each expert sequentially. This approach is computationally inefficient, especially when dealing with a large number of experts.

    Args:
        token_states (torch.Tensor): Input tensor of shape (num_tokens, in_features).
        expert_weights (torch.Tensor): Weight tensor of shape (num_experts, in_features, out_features).
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

    Returns:
        torch.Tensor: Output tensor of shape (num_tokens, out_features).
    """
    num_tokens = token_states.shape[0]
    out_features = expert_weights.shape[-1]
    output = torch.zeros(num_tokens, out_features, dtype=token_states.dtype, device=token_states.device)

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    for expert_num in range(expert_weights.shape[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        tokens = token_states[start:end]

        out = torch.matmul(tokens, expert_weights[expert_num])
        output[start:end] = out
    return output


class AriaTextConfig(LlamaConfig):
    r"""
    This class handles the configuration for the text component of the Aria model.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the Aria
    [rhymes-ai/Aria](https://huggingface.co/rhymes-ai/Aria) architecture.
    This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4096):
            The size of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 2):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_heads
        moe_num_experts (`int`, *optional*, defaults to 8):
            The number of experts in the MoE layer.
        moe_topk (`int`, *optional*, defaults to 2):
            The number of top experts to route to for each token.
        moe_num_shared_experts (`int`, *optional*, defaults to 2):
            The number of shared experts.
    """

    model_type = "aria_text"
    base_config_key = "text_config"

    def __init__(
        self,
        intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_num_shared_experts: int = 2,
        pad_token_id=2,
        **super_kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **super_kwargs)
        self.intermediate_size = intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_num_shared_experts = moe_num_shared_experts


class AriaConfig(PretrainedConfig):
    r"""
    This class handles the configuration for both vision and text components of the Aria model,
    as well as additional parameters for image token handling and projector mapping.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the Aria
    [rhymes-ai/Aria](https://huggingface.co/rhymes-ai/Aria) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`AriaVisionConfig` or `dict`, *optional*):
            Configuration for the vision component.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to select the vision feature.
        text_config (`AriaTextConfig` or `dict`, *optional*):
            Configuration for the text component.
        projector_patch_to_query_dict (`dict`, *optional*):
            Mapping of patch sizes to query dimensions.
        image_token_index (`int`, *optional*, defaults to 9):
            Index used to represent image tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.

    Attributes:
        model_type (`str`):
            Type of the model, set to `"aria"`.
        image_token_index (`int`):
            Index used to represent image tokens.
        projector_patch_to_query_dict (`dict`):
            Mapping of patch sizes to query dimensions.
        vision_config (`AriaVisionConfig`):
            Configuration for the vision component.
        text_config (`AriaTextConfig`):
            Configuration for the text component.
    """

    model_type = "aria"
    sub_configs = {"text_config": AriaTextConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        vision_feature_layer: int = -1,
        text_config: AriaTextConfig = None,
        projector_patch_to_query_dict: Dict = None,
        image_token_index: int = 9,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.image_token_index = image_token_index

        # Convert the keys and values of projector_patch_to_query_dict to integers
        # This ensures consistency even if they were provided as strings
        if projector_patch_to_query_dict is None:
            projector_patch_to_query_dict = {
                1225: 128,
                4900: 256,
            }
        self.projector_patch_to_query_dict = {int(k): int(v) for k, v in projector_patch_to_query_dict.items()}
        self.max_value_projector_patch_to_query_dict = max(self.projector_patch_to_query_dict.values())
        self.vision_feature_layer = vision_feature_layer
        if isinstance(vision_config, dict):
            vision_config["model_type"] = "idefics3_vision"
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["idefics3_vision"]()

        self.vision_config = vision_config
        self.initializer_range = initializer_range

        if isinstance(text_config, dict) and "model_type" in text_config:
            text_config = AriaTextConfig(**text_config)
        elif text_config is None:
            text_config = AriaTextConfig()

        self.text_config = text_config

        super().__init__(**kwargs)


class AriaTextRMSNorm(LlamaRMSNorm):
    pass


class AriaProjectorMLP(nn.Module):
    """
    Feed-Forward Network module for the Aria Projector.

    Args:
        in_features (`int`):
            Input embedding dimension.
        hidden_features (`int`):
            Hidden dimension of the feed-forward network.
        output_dim (`int`):
            Output dimension.
    """

    def __init__(self, in_features, hidden_features, output_dim):
        super().__init__()
        self.linear_in = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_out = nn.Linear(hidden_features, output_dim, bias=False)
        self.act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_states = self.act(self.linear_in(hidden_states))
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class AriaCrossAttention(nn.Module):
    """
    Aria Cross-Attention module.

    Args:
        config (`AriaConfig`):
            The configuration to use.
    """

    def __init__(self, config: AriaConfig, dropout_rate: float = 0):
        super().__init__()
        hidden_size = config.vision_config.hidden_size
        num_heads = config.vision_config.num_attention_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Original code here: https://github.com/rhymes-ai/Aria/blob/719ff4e52b727443cba3793b0e27fe64e0244fe1/aria/model/projector.py#L48
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm_kv = nn.LayerNorm(hidden_size)

    def forward(self, key_value_states, hidden_states, attn_mask=None):
        """
        Forward pass of the AriaCrossAttention module.

        Args:
            key_value_states (`torch.Tensor`):
                Input tensor for key and value.
            hidden_states (`torch.Tensor`):
                Input tensor for query.
            attn_mask (`torch.Tensor`, *optional*, defaults to None):
                Attention mask.

        Returns:
            torch.Tensor:
                Output tensor after cross-attention.
        """
        query = self.q_proj(self.layer_norm(hidden_states))

        key_value_states = self.layer_norm_kv(key_value_states)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)

        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)

        attn_output = self.dropout(self.linear(attn_output))

        return attn_output


class AriaProjector(nn.Module):
    """
    Aria Projector module.

    This module projects vision features into the language model's embedding space, enabling interaction between vision and language components.

    Args:
        config (`AriaConfig`):
            Configuration object for the model.
    """

    def __init__(
        self,
        config: AriaConfig,
    ):
        super().__init__()

        self.patch_to_query_dict = config.projector_patch_to_query_dict
        self.in_features = config.vision_config.hidden_size
        self.num_heads = config.vision_config.num_attention_heads
        self.kv_dim = config.vision_config.hidden_size
        self.hidden_features = config.text_config.hidden_size
        self.output_dim = config.text_config.hidden_size

        self.query = nn.Parameter(torch.zeros(config.max_value_projector_patch_to_query_dict, self.in_features))

        self.cross_attn = AriaCrossAttention(config)

        self.layer_norm = nn.LayerNorm(self.in_features)
        self.feed_forward = AriaProjectorMLP(self.in_features, self.hidden_features, self.output_dim)

    def forward(self, key_value_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the Projector module.

        Args:
            key_value_states (`torch.Tensor`):
                Input tensor of shape (batch_size, num_patches, kv_dim).
            attn_mask (`torch.Tensor`, *optional*, default is None):
                Attention mask.

        Returns:
            `torch.Tensor`: Output tensor of shape (batch_size, query_number, output_dim).
        """
        batch_size, num_patches = key_value_states.shape[0], key_value_states.shape[1]

        if num_patches not in self.patch_to_query_dict.keys():
            raise KeyError(
                f"Number of patches {num_patches} not found in patch_to_query_dict amongst possible values {self.patch_to_query_dict.keys()}."
            )
        query_num = self.patch_to_query_dict[num_patches]

        queries = self.query[:query_num].unsqueeze(0).repeat(batch_size, 1, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(key_value_states, queries, attn_mask=attn_mask)

        out = self.feed_forward(self.layer_norm(attention_out))

        return out


def _get_patch_output_size(image, target_resolution, input_data_format):
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


class AriaImageProcessor(BaseImageProcessor):
    """
    A vision processor for the Aria model that handles image preprocessing.
    Initialize the AriaImageProcessor.

    Args:
        image_mean (`list`, *optional*, defaults to [0.5, 0.5, 0.5]):
            Mean values for normalization.
        image_std (`list`, *optional*, defaults to [0.5, 0.5, 0.5]):
            Standard deviation values for normalization.
        max_image_size (`int`, *optional*, defaults to 980):
            Maximum image size.
        min_image_size (`int`, *optional*, defaults to 336):
            Minimum image size.
        split_resolutions (`list`, *optional*, defaults to a list of optimal,resolutions as tuples):
            The optimal resolutions for splitting the image.
        split_image (`bool`, *optional*, defaults to `False`):
            Whether to split the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        resample (PILImageResampling, *optional*, defaults to `BICUBIC`):
            The resampling filter to use if resizing the image.
    """

    def __init__(
        self,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        max_image_size: int = 980,
        min_image_size: int = 336,
        split_resolutions: Optional[List[Tuple[int, int]]] = None,
        split_image: Optional[bool] = False,
        do_convert_rgb: Optional[bool] = True,
        do_normalize: Optional[bool] = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if image_mean is None:
            image_mean = [0.5, 0.5, 0.5]
        if image_std is None:
            image_std = [0.5, 0.5, 0.5]
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.split_image = split_image
        if split_resolutions is None:
            split_resolutions = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1), (7, 1), (8, 1)]  # fmt: skip
            split_resolutions = [(el[0] * 490, el[1] * 490) for el in split_resolutions]
        self.split_resolutions = split_resolutions
        self.do_convert_rgb = do_convert_rgb
        self.do_normalize = do_normalize
        self.resample = resample

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        max_image_size: Optional[int] = None,
        min_image_size: Optional[int] = None,
        split_image: Optional[bool] = None,
        do_convert_rgb: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        resample: PILImageResampling = None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Process a list of images.

        Args:
            images (ImageInput or list of ImageInput):
                The input image or a list of images.
            image_mean (`list`, *optional*, defaults to [0.5, 0.5, 0.5]):
                Mean values for normalization.
            image_std (`list`, *optional*, defaults to [0.5, 0.5, 0.5]):
                Standard deviation values for normalization.
            max_image_size (`int`, *optional*, defaults to `self.max_image_size` (980)):
                Maximum image size.
            min_image_size (`int`, *optional*, defaults to `self.min_image_size` (336)):
                Minimum image size.
            split_image (`bool`, *optional*, defaults to `self.split_image` (False)):
                Whether to split the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb` (True)):
                Whether to convert the image to RGB.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize` (True)):
                Whether to normalize the image.
            resample (PILImageResampling, *optional*, defaults to `self.resample` (BICUBIC)):
                The resampling filter to use if resizing the image.
            return_tensors (`str` or `TensorType`, *optional*, defaults to "pt"):
                The type of tensor to return.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`:
                        image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`:
                        image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`:
                        image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`:
                        image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            BatchFeature:
                A BatchFeature object containing:
                - 'pixel_values':
                    Tensor of processed image pixel values.
                - 'pixel_mask':
                    Boolean pixel mask. This mask is a 2D tensor of shape (max_image_size, max_image_size) where:
                    - True (1) values indicate pixels that belong to the original resized image.
                    - False (0) values indicate pixels that are part of the padding.
                  The mask helps distinguish between actual image content and padded areas in subsequent processing steps.
                - 'num_crops':
                    The maximum number of crops across all images.
        """
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        max_image_size = max_image_size if max_image_size is not None else self.max_image_size
        min_image_size = min_image_size if min_image_size is not None else self.min_image_size
        split_image = split_image if split_image is not None else self.split_image
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample

        if max_image_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")

        images = make_batched_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        pixel_values = []
        pixel_masks = []
        num_crops = None

        for image in images:
            if split_image:
                crop_images = self.get_image_patches(
                    image,
                    self.split_resolutions,
                    max_image_size,
                    resample,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )
            else:
                crop_images = [image]
            if num_crops is None or len(crop_images) > num_crops:
                num_crops = len(crop_images)

            for crop_image in crop_images:
                # At this point the scale is the rescaling factor that would bring the image to max_size in its larger dimension
                h, w = get_image_size(crop_image)
                scale = max_image_size / max(h, w)
                if w >= h:
                    new_size = (max(int(h * scale), min_image_size), max_image_size)  # h, w
                else:
                    new_size = (max_image_size, max(int(w * scale), min_image_size))  # h, w

                crop_image_resized = resize(
                    crop_image,
                    new_size,
                    resample=resample,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )

                padding_bottom, padding_right = max_image_size - new_size[0], max_image_size - new_size[1]
                crop_image_padded = pad(
                    crop_image_resized,
                    ((0, padding_bottom), (0, padding_right)),
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )

                # Create a pixel mask
                pixel_mask = np.zeros((max_image_size, max_image_size), dtype=bool)
                pixel_mask[: new_size[0], : new_size[1]] = 1
                pixel_masks.append(pixel_mask)

                if do_normalize:
                    crop_image_padded = self.normalize(
                        crop_image_padded / 255.0,
                        self.image_mean,
                        self.image_std,
                        data_format=input_data_format,
                        input_data_format=input_data_format,
                    )
                    crop_image_padded = (
                        to_channel_dimension_format(crop_image_padded, data_format, input_data_format)
                        if data_format is not None
                        else crop_image_padded
                    )

                pixel_values.append(crop_image_padded)
        return BatchFeature(
            data={
                "pixel_values": np.stack(pixel_values, axis=0),
                "pixel_mask": np.stack(pixel_masks, axis=0),
                "num_crops": num_crops,
            },
            tensor_type=return_tensors,
        )

    def _resize_for_patching(
        self, image: np.array, target_resolution: tuple, resample, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        """
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = resize(image, (new_height, new_width), resample=resample, input_data_format=input_data_format)

        return resized_image

    def _pad_for_patching(
        self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = self.pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        return padded_image

    def pad(
        self,
        image: np.ndarray,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
        dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
        as input.

        Args:
            image (`np.ndarray`):
                The image to pad.
            padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
                Padding to apply to the edges of the height, width axes. Can be one of three formats:
                - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
                - `((before, after),)` yields same before and after pad for height and width.
                - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            `np.ndarray`: The padded image.

        """

        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values, data_format, input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        padding_mode_mapping = {
            PaddingMode.CONSTANT: "constant",
            PaddingMode.REFLECT: "reflect",
            PaddingMode.REPLICATE: "edge",
            PaddingMode.SYMMETRIC: "symmetric",
        }
        image = np.pad(image, padding, mode=padding_mode_mapping[mode], constant_values=constant_values)
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )
        return image

    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints: List[Tuple[int, int]],
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> List[np.array]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (`np.array`):
                The input image to be processed.
            grid_pinpoints (List[Tuple[int, int]]):
                A list of possible resolutions as tuples.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            `List[np.array]`: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)

        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]
        return patches


class AriaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "max_image_size": 980,
            "split_image": False,
        },
        "return_tensors": TensorType.PYTORCH,
    }


class AriaProcessor(ProcessorMixin):
    """
    AriaProcessor is a processor for the Aria model which wraps the Aria image preprocessor and the LLama slow tokenizer.

    Args:
        image_processor (`AriaImageProcessor`, *optional*):
            The AriaImageProcessor to use for image preprocessing.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
        size_conversion (`Dict`, *optional*):
            A dictionary indicating size conversions for images.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "size_conversion"]
    image_processor_class = "AriaImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer: Union[AutoTokenizer, str] = None,
        chat_template: Optional[str] = None,
        size_conversion: Optional[Dict[Union[float, int], int]] = None,
    ):
        if size_conversion is None:
            size_conversion = {490: 128, 980: 256}
        self.size_conversion = {int(k): v for k, v in size_conversion.items()}

        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Optional[ImageInput] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[AriaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s).

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`ImageInput`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.


        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_mask** -- Pixel mask to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            AriaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        if images is not None:
            image_inputs = self.image_processor(
                images,
                **output_kwargs["images_kwargs"],
            )
            # expand the image_token according to the num_crops and tokens per image
            tokens_per_image = self.size_conversion[image_inputs.pixel_values.shape[2]]
            prompt_strings = []
            num_crops = image_inputs.pop("num_crops") * tokens_per_image
            for sample in text:
                sample = sample.replace(self.tokenizer.image_token, self.tokenizer.image_token * num_crops)
                prompt_strings.append(sample)

        else:
            image_inputs = {}
            prompt_strings = text

        text_inputs = self.tokenizer(
            prompt_strings,
            **output_kwargs["text_kwargs"],
        )

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class AriaSharedExpertsMLP(LlamaMLP):
    """
    Shared Expert MLP for shared experts.

    Unlike routed experts, shared experts process all tokens without routing.
    This class reconfigures the intermediate size in comparison to the LlamaMLP.

    Args:
        config (`AriaTextConfig`): Configuration object for the Aria language model.
    """

    def __init__(self, config: AriaTextConfig):
        super().__init__(self)
        self.intermediate_size = config.intermediate_size * config.moe_num_shared_experts


class AriaGroupedExpertsGemm(nn.Module):
    """
    Grouped GEMM (General Matrix Multiplication) module for efficient expert computation.
    This module utilizes the grouped_gemm library (https://github.com/fanshiqing/grouped_gemm)
    for optimized performance. If the grouped_gemm library is not installed, it gracefully
    falls back to a sequential GEMM implementation, which may be slower but ensures
    functionality.

    Args:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        groups (`int`):
            Number of expert groups.
    """

    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features))

    def forward(self, input, tokens_per_expert):
        """
        Perform grouped matrix multiplication.

        Args:
            input (`torch.Tensor`):
                Input tensor of shape (num_tokens, in_features).
            tokens_per_expert (`torch.Tensor`):
                Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor of shape (num_tokens, out_features).
        """
        return sequential_experts_gemm(
            input,
            self.weight,
            tokens_per_expert.cpu(),
        )


class AriaGroupedExpertsMLP(nn.Module):
    """
    Grouped MLP module for Mixture of Experts.

    Args:
        config (`AriaTextConfig`):
            Configuration object for the model.
    """

    def __init__(self, config: AriaTextConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = AriaGroupedExpertsGemm(config.hidden_size, config.intermediate_size * 2, config.moe_num_experts)
        self.fc2 = AriaGroupedExpertsGemm(config.intermediate_size, config.hidden_size, config.moe_num_experts)

    def forward(self, permuted_tokens, tokens_per_expert):
        """
        Forward pass of the Grouped MLP.

        Args:
            permuted_tokens (torch.Tensor): Permuted input tokens.
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        fc1_output = self.fc1(permuted_tokens, tokens_per_expert)
        projection, gate = torch.chunk(fc1_output, 2, dim=-1)
        fc1_output = nn.functional.silu(projection) * gate
        fc2_output = self.fc2(fc1_output, tokens_per_expert)
        return fc2_output


# Token permutation adapted from https://github.com/NVIDIA/Megatron-LM/blob/54f1f78529cbc2b9cddad313e7f9d96ac0420a27/megatron/core/transformer/moe/token_dispatcher.py#L291-L587
class AriaTextMoELayer(nn.Module):
    """
    Aria Text Mixture of Experts (MoE) Layer.

    This layer applies a gating mechanism to route input tokens to different experts.

    Args:
        config (`AriaTextConfig`):
            Configuration object for the text component of the model.
    """

    def __init__(self, config: AriaTextConfig):
        super().__init__()

        self.router = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.experts = AriaGroupedExpertsMLP(config)
        self.shared_experts = AriaSharedExpertsMLP(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE Layer.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.

        Process:
        1. Route tokens to experts using the router.
        2. Permute tokens based on routing decisions.
        3. Process tokens through experts.
        4. Unpermute and combine expert outputs.
        5. Add shared expert output to the final result.
        """
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        # Top K Routing
        logits = self.router(hidden_states)
        top_logits, top_indices = torch.topk(logits, k=self.config.moe_topk, dim=1)
        scores = nn.functional.softmax(top_logits, dim=-1)

        original_dtype = top_indices.dtype

        tokens_per_expert = torch.histc(
            top_indices.flatten().to(torch.float32),
            bins=self.config.moe_num_experts,
            min=0,
            max=self.config.moe_num_experts - 1,
        ).to(original_dtype)
        indices = top_indices

        # Token permutation
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices)
        permuted_tokens = hidden_states.index_select(0, sorted_indices // self.config.moe_topk)

        # Process through experts
        expert_output = self.experts(permuted_tokens, tokens_per_expert)

        # Token unpermutation
        unpermuted_tokens = torch.zeros(
            (scores.shape[0] * self.config.moe_topk, expert_output.size(1)),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unpermuted_tokens.index_copy_(0, sorted_indices, expert_output)
        unpermuted_tokens = unpermuted_tokens.view(-1, self.config.moe_topk, expert_output.size(1))

        output = (unpermuted_tokens * scores.unsqueeze(-1)).sum(dim=1).view(original_shape)

        # Add shared expert output
        shared_expert_output = self.shared_experts(hidden_states.view(original_shape))
        return output + shared_expert_output


class AriaTextDecoderLayer(LlamaDecoderLayer):
    """
    Aria Text Decoder Layer.

    This class defines a single decoder layer in the language model, incorporating self-attention and Mixture of Experts (MoE) feed-forward network.

    Args:
        config (`AriaTextConfig`):
            Configuration object for the text component of the model.
        layer_idx (`int`):
            Index of the layer.
    """

    def __init__(self, config: AriaTextConfig, layer_idx: int):
        super().__init__(self)
        self.mlp = AriaTextMoELayer(config)


class AriaTextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = AriaConfig
    base_model_prefix = "model"
    _no_split_modules = ["AriaTextDecoderLayer", "AriaGroupedExpertsGemm"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AriaGroupedExpertsGemm):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


class AriaPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AriaProjector):
            nn.init.trunc_normal_(module.query, std=std)


class AriaTextModel(LlamaModel):
    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AriaTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()


class AriaTextForCausalLM(AriaTextPreTrainedModel, LlamaForCausalLM):
    """
    Aria model for causal language modeling tasks.

    This class extends `LlamaForCausalLM` to incorporate the Mixture of Experts (MoE) approach,
    allowing for more efficient and scalable language modeling.

    Args:
        config (`AriaTextConfig`):
            Configuration object for the model.
    """

    _tied_weights_keys = ["lm_head.weight"]
    config_class = AriaTextConfig

    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.model = AriaTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class AriaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


ARIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`, *optional*):
            Input token IDs.
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of the images.
        pixel_mask (`torch.LongTensor`, *optional*):
            Mask for the pixel values.
        attention_mask (`torch.Tensor`, *optional*):
            Attention mask.
        position_ids (`torch.LongTensor`, *optional*):
            Position IDs.
        past_key_values (`List[torch.FloatTensor]`, *optional*):
            Past key values for efficient processing.
        inputs_embeds (`torch.FloatTensor`, *optional*):
            Input embeddings.
        labels (`torch.LongTensor`, *optional*):
            Labels for computing the language modeling loss.
        use_cache (`bool`, *optional*):
            Whether to use the model's cache mechanism.
        output_attentions (`bool`, *optional*):
            Whether to output attention weights.
        output_hidden_states (`bool`, *optional*):
            Whether to output hidden states.
        return_dict (`bool`, *optional*):
            Whether to return a `ModelOutput` object.
        num_logits_to_keep (`int`, *optional*, defaults to 0):
            Calculate logits for the last `num_logits_to_keep` tokens, or all `input_ids` if `0`.
        cache_position (`torch.LongTensor`, *optional*):
            Cache positions.
        **loss_kwargs:
            Additional keyword arguments for loss calculation.
"""

ARIA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (`AriaConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language model
    to perform tasks that involve both image and text inputs.""",
    ARIA_START_DOCSTRING,
)
class AriaForConditionalGeneration(AriaPreTrainedModel, GenerationMixin):
    config_class = AriaConfig
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _tied_weights_keys = ["language_model.lm_head.weight"]

    def __init__(self, config: AriaConfig):
        super().__init__(config)

        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = AriaProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2"
        self.post_init()

    def _create_patch_attention_mask(self, pixel_mask):
        if pixel_mask is None:
            return None

        patches_subgrid = pixel_mask.unfold(
            dimension=1,
            size=self.vision_tower.config.patch_size,
            step=self.vision_tower.config.patch_size,
        )
        patches_subgrid = patches_subgrid.unfold(
            dimension=2,
            size=self.vision_tower.config.patch_size,
            step=self.vision_tower.config.patch_size,
        )
        return (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor = None,
        vision_feature_layer: int = -1,
    ):
        patch_attention_mask = self._create_patch_attention_mask(pixel_mask)
        image_outputs = self.vision_tower(
            pixel_values, patch_attention_mask=patch_attention_mask, output_hidden_states=True
        )
        image_attn_mask = None
        if patch_attention_mask is not None:
            flattened_mask = patch_attention_mask.flatten(1)
            image_attn_mask = torch.logical_not(flattened_mask)

        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        image_features = self.multi_modal_projector(selected_image_feature, attn_mask=image_attn_mask)
        return image_features

    @add_start_docstrings_to_model_forward(ARIA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AriaCausalLMOutputWithPast, config_class=AriaConfig)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        cache_position: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, AriaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics3ForConditionalGeneration`).
                Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
                computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:

        Example:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModel
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("Rhymes-AI/Aria")
        >>> model = AutoModel.from_pretrained("Rhymes-AI/Aria", torch_dtype=torch.bfloat16, device_map="auto")

        >>> # Create inputs
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ...             {"type": "image"},
        ...             {"type": "text", "text": "What can we see in this image?"},
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "In which city is that bridge located?"},
        ...         ]
        ...     }
        ... ]

        >>> prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
        >>> images = [[image1, image2], [image3]]
        >>> inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts[0])
        Assistant: There are buildings, trees, lights, and water visible in this image.

        >>> print(generated_texts[1])
        Assistant: The bridge is in San Francisco.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and inputs_embeds.shape[1] != 1:
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
                n_image_tokens = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            else:
                image_embeds = input_ids == self.config.image_token_index
                special_image_mask = image_embeds.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                n_image_tokens = (image_embeds).sum(dim=1).sum(dim=0)
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                vision_feature_layer=self.config.vision_feature_layer,
            )
            n_images, n_features_per_image = image_features.shape[0], image_features.shape[1]
            n_image_features = n_images * n_features_per_image
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **loss_kwargs
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AriaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_mask=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_mask"] = pixel_mask

        return model_inputs


__all__ = [
    "AriaConfig",
    "AriaTextConfig",
    "AriaImageProcessor",
    "AriaProcessor",
    "AriaForConditionalGeneration",
    "AriaPreTrainedModel",
    "AriaTextPreTrainedModel",
    "AriaTextModel",
    "AriaTextForCausalLM",
]
