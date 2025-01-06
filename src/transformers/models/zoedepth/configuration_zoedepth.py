# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""ZoeDepth model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/zoedepth-nyu": "https://huggingface.co/Intel/zoedepth-nyu/resolve/main/config.json",
}


class ZoeDepthConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ZoeDepthForDepthEstimation`]. It is used to instantiate an ZoeDepth
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ZoeDepth
    [Intel/zoedepth-nyu](https://huggingface.co/Intel/zoedepth-nyu) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*, defaults to `BeitConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        readout_type (`str`, *optional*, defaults to `"project"`):
            The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
            the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].

            - "ignore" simply ignores the CLS token.
            - "add" passes the information from the CLS token to all other tokens by adding the representations.
            - "project" passes information to the other tokens by concatenating the readout to all other tokens before
              projecting the
            representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
        reassemble_factors (`List[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`List[str]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the heads.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the pre-activate residual units of the fusion blocks.
        num_relative_features (`int`, *optional*, defaults to 32):
            The number of features to use in the relative depth estimation head.
        add_projection (`bool`, *optional*, defaults to `False`):
            Whether to add a projection layer before the depth estimation head.
        bottleneck_features (`int`, *optional*, defaults to 256):
            The number of features in the bottleneck layer.
        num_attractors (`List[int], *optional*, defaults to `[16, 8, 4, 1]`):
            The number of attractors to use in each stage.
        bin_embedding_dim (`int`, *optional*, defaults to 128):
            The dimension of the bin embeddings.
        attractor_alpha (`int`, *optional*, defaults to 1000):
            The alpha value to use in the attractor.
        attractor_gamma (`int`, *optional*, defaults to 2):
            The gamma value to use in the attractor.
        attractor_kind (`str`, *optional*, defaults to `"mean"`):
            The kind of attractor to use. Can be one of [`"mean"`, `"sum"`].
        min_temp (`float`, *optional*, defaults to 0.0212):
            The minimum temperature value to consider.
        max_temp (`float`, *optional*, defaults to 50.0):
            The maximum temperature value to consider.
        bin_centers_type (`str`, *optional*, defaults to `"softplus"`):
            Activation type used for bin centers. Can be "normed" or "softplus". For "normed" bin centers, linear normalization trick
            is applied. This results in bounded bin centers. For "softplus", softplus activation is used and thus are unbounded.
        bin_configurations (`List[dict]`, *optional*, defaults to `[{'n_bins': 64, 'min_depth': 0.001, 'max_depth': 10.0}]`):
            Configuration for each of the bin heads.
            Each configuration should consist of the following keys:
            - name (`str`): The name of the bin head - only required in case of multiple bin configurations.
            - `n_bins` (`int`): The number of bins to use.
            - `min_depth` (`float`): The minimum depth value to consider.
            - `max_depth` (`float`): The maximum depth value to consider.
            In case only a single configuration is passed, the model will use a single head with the specified configuration.
            In case multiple configurations are passed, the model will use multiple heads with the specified configurations.
        num_patch_transformer_layers (`int`, *optional*):
            The number of transformer layers to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_hidden_size (`int`, *optional*):
            The hidden size to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_intermediate_size (`int`, *optional*):
            The intermediate size to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_num_attention_heads (`int`, *optional*):
            The number of attention heads to use in the patch transformer. Only used in case of multiple bin configurations.

    Example:

    ```python
    >>> from transformers import ZoeDepthConfig, ZoeDepthForDepthEstimation

    >>> # Initializing a ZoeDepth zoedepth-large style configuration
    >>> configuration = ZoeDepthConfig()

    >>> # Initializing a model from the zoedepth-large style configuration
    >>> model = ZoeDepthForDepthEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "zoedepth"

    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        backbone_kwargs=None,
        hidden_act="gelu",
        initializer_range=0.02,
        batch_norm_eps=1e-05,
        readout_type="project",
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[96, 192, 384, 768],
        fusion_hidden_size=256,
        head_in_index=-1,
        use_batch_norm_in_fusion_residual=False,
        use_bias_in_fusion_residual=None,
        num_relative_features=32,
        add_projection=False,
        bottleneck_features=256,
        num_attractors=[16, 8, 4, 1],
        bin_embedding_dim=128,
        attractor_alpha=1000,
        attractor_gamma=2,
        attractor_kind="mean",
        min_temp=0.0212,
        max_temp=50.0,
        bin_centers_type="softplus",
        bin_configurations=[{"n_bins": 64, "min_depth": 0.001, "max_depth": 10.0}],
        num_patch_transformer_layers=None,
        patch_transformer_hidden_size=None,
        patch_transformer_intermediate_size=None,
        patch_transformer_num_attention_heads=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if readout_type not in ["ignore", "add", "project"]:
            raise ValueError("Readout_type must be one of ['ignore', 'add', 'project']")

        if attractor_kind not in ["mean", "sum"]:
            raise ValueError("Attractor_kind must be one of ['mean', 'sum']")

        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `BEiT` backbone.")
            backbone_config = CONFIG_MAPPING["beit"](
                image_size=384,
                num_hidden_layers=24,
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                use_relative_position_bias=True,
                reshape_hidden_states=False,
                out_features=["stage6", "stage12", "stage18", "stage24"],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.hidden_act = hidden_act
        self.use_pretrained_backbone = use_pretrained_backbone
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.readout_type = readout_type
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.use_batch_norm_in_fusion_residual = use_batch_norm_in_fusion_residual
        self.use_bias_in_fusion_residual = use_bias_in_fusion_residual
        self.num_relative_features = num_relative_features
        self.add_projection = add_projection

        self.bottleneck_features = bottleneck_features
        self.num_attractors = num_attractors
        self.bin_embedding_dim = bin_embedding_dim
        self.attractor_alpha = attractor_alpha
        self.attractor_gamma = attractor_gamma
        self.attractor_kind = attractor_kind
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.bin_centers_type = bin_centers_type
        self.bin_configurations = bin_configurations
        self.num_patch_transformer_layers = num_patch_transformer_layers
        self.patch_transformer_hidden_size = patch_transformer_hidden_size
        self.patch_transformer_intermediate_size = patch_transformer_intermediate_size
        self.patch_transformer_num_attention_heads = patch_transformer_num_attention_heads


__all__ = ["ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP", "ZoeDepthConfig"]
