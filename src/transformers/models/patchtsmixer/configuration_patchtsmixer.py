# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All Rights Reserved.
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
"""PatchTSMixer model configuration"""

from typing import List, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class PatchTSMixerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PatchTSMixerModel`]. It is used to instantiate a
    PatchTSMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PatchTSMixer
    [ibm/patchtsmixer-etth1-pretrain](https://huggingface.co/ibm/patchtsmixer-etth1-pretrain) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 32):
            The context/history length for the input sequence.
        patch_length (`int`, *optional*, defaults to 8):
            The patch length for the input sequence.
        num_input_channels (`int`, *optional*, defaults to 1):
            Number of input variates. For Univariate, set it to 1.
        patch_stride (`int`, *optional*, defaults to 8):
            Determines the overlap between two consecutive patches. Set it to patch_length (or greater), if we want
            non-overlapping patches.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for probabilistic forecast.
        d_model (`int`, *optional*, defaults to 8):
            Hidden dimension of the model. Recommended to set it as a multiple of patch_length (i.e. 2-5X of
            patch_length). Larger value indicates more complex model.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 3):
            Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` backbone. Recommended range is 0.2-0.7
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Mixer Mode. Determines how to process the channels. Allowed values: "common_channel", "mix_channel". In
            "common_channel" mode, we follow Channel-independent modelling with no explicit channel-mixing. Channel
            mixing happens in an implicit manner via shared weights across channels. (preferred first approach) In
            "mix_channel" mode, we follow explicit channel-mixing in addition to patch and feature mixer. (preferred
            approach when channel correlations are very important to model)
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention across patches. This can be enabled when the output of Vanilla PatchTSMixer with
            gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
            across patches.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
            set to `True`.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. Options `"random"` and `"sincos"` are supported. Works only when
            `use_positional_encoding` is set to `True`
        scaling (`string` or `bool`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        loss (`string`, *optional*, defaults to `"mse"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood ("nll") and for point estimates it is the mean squared
            error "mse".
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, or the default initialization in
            `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        mask_type (`str`, *optional*, defaults to `"random"`):
            Type of masking to use for Masked Pretraining mode. Allowed values are "random", "forecast". In Random
            masking, points are masked randomly. In Forecast masking, points are masked towards the end.
        random_mask_ratio (`float`, *optional*, defaults to 0.5):
            Masking ratio to use when `mask_type` is `random`. Higher value indicates more masking.
        num_forecast_mask_patches (`int` or `list`, *optional*, defaults to `[2]`):
            Number of patches to be masked at the end of each batch sample. If it is an integer, all the samples in the
            batch will have the same number of masked patches. If it is a list, samples in the batch will be randomly
            masked by numbers defined in the list. This argument is only used for forecast pretraining.
        mask_value (`float`, *optional*, defaults to `0.0`):
            Mask value to use.
        masked_loss (`bool`, *optional*, defaults to `True`):
            Whether to compute pretraining loss only at the masked portions, or on the entire output.
        channel_consistent_masking (`bool`, *optional*, defaults to `True`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        unmasked_channel_indices (`list`, *optional*):
            Channels that are not masked during pretraining.
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` head.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model when loss is "nll". Could be either "student_t", "normal" or
            "negative_binomial".
        prediction_length (`int`, *optional*, defaults to 16):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
        prediction_channel_indices (`list`, *optional*):
            List of channel indices to forecast. If None, forecast all channels. Target data is expected to have all
            channels and we explicitly filter the channels in prediction and target before loss computation.
        num_targets (`int`, *optional*, defaults to 3):
            Number of targets (dimensionality of the regressed variable) for a regression task.
        output_range (`list`, *optional*):
            Output range to restrict for the regression task. Defaults to None.
        head_aggregation (`str`, *optional*, defaults to `"max_pool"`):
            Aggregation mode to enable for classification or regression task. Allowed values are `None`, "use_last",
            "max_pool", "avg_pool".

    Example:

    ```python
    >>> from transformers import PatchTSMixerConfig, PatchTSMixerModel

    >>> # Initializing a default PatchTSMixer configuration
    >>> configuration = PatchTSMixerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSMixerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "patchtsmixer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 32,
        patch_length: int = 8,
        num_input_channels: int = 1,
        patch_stride: int = 8,
        num_parallel_samples: int = 100,
        # General model configuration
        d_model: int = 8,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: Optional[Union[str, bool]] = "std",
        loss: str = "mse",
        init_std: float = 0.02,
        post_init: bool = False,
        norm_eps: float = 1e-5,
        # Pretrain model configuration
        mask_type: str = "random",
        random_mask_ratio: float = 0.5,
        num_forecast_mask_patches: Optional[Union[List[int], int]] = [2],
        mask_value: int = 0,
        masked_loss: bool = True,
        channel_consistent_masking: bool = True,
        unmasked_channel_indices: Optional[List[int]] = None,
        # General head configuration
        head_dropout: float = 0.2,
        distribution_output: str = "student_t",
        # Prediction head configuration
        prediction_length: int = 16,
        prediction_channel_indices: list = None,
        # Classification/Regression configuration
        num_targets: int = 3,
        output_range: list = None,
        head_aggregation: str = "max_pool",
        **kwargs,
    ):
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
        self.mask_type = mask_type
        self.random_mask_ratio = random_mask_ratio
        self.num_forecast_mask_patches = num_forecast_mask_patches
        self.mask_value = mask_value
        self.channel_consistent_masking = channel_consistent_masking
        self.masked_loss = masked_loss
        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.prediction_length = prediction_length
        self.prediction_channel_indices = prediction_channel_indices
        self.num_targets = num_targets
        self.output_range = output_range
        self.head_aggregation = head_aggregation
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.distribution_output = distribution_output
        self.loss = loss
        self.num_parallel_samples = num_parallel_samples
        self.unmasked_channel_indices = unmasked_channel_indices
        self.norm_eps = norm_eps
        super().__init__(**kwargs)


__all__ = ["PatchTSMixerConfig"]
