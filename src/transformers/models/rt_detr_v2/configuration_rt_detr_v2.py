#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/rt_detr_v2/modular_rt_detr_v2.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_rt_detr_v2.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2024 Baidu Inc and The HuggingFace Inc. team.
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

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import (
    BackboneConfigMixin,
    get_aligned_output_features_output_indices,
    verify_backbone_config_arguments,
)
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class RtDetrV2ResNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RtDetrV2ResnetBackbone`]. It is used to instantiate an
    ResNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ResNet
    [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            The layer to use, it can be either `"basic"` (used for smaller models, like resnet-18 or resnet-34) or
            `"bottleneck"` (used for larger models like resnet-50 and above).
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.
        downsample_in_bottleneck (`bool`, *optional*, defaults to `False`):
            If `True`, the first conv 1x1 in ResNetBottleNeckLayer will downsample the inputs using a `stride` of 2.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.

    Example:
    ```python
    >>> from transformers import RtDetrV2ResNetConfig, RtDetrV2ResnetBackbone

    >>> # Initializing a ResNet resnet-50 style configuration
    >>> configuration = RtDetrV2ResNetConfig()

    >>> # Initializing a model (with random weights) from the resnet-50 style configuration
    >>> model = RtDetrV2ResnetBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "rt_detr_v2_resnet"
    layer_types = ["basic", "bottleneck"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="bottleneck",
        hidden_act="relu",
        downsample_in_first_stage=False,
        downsample_in_bottleneck=False,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage
        self.downsample_in_bottleneck = downsample_in_bottleneck
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )


class RtDetrV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RtDetrV2Model`]. It is used to instantiate a
    RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RT-DETR
    [checkpoing/todo](https://huggingface.co/checkpoing/todo) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_bias_prior_prob (`float`, *optional*):
            The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
            If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        backbone_config (`Dict`, *optional*, defaults to `RtDetrV2ResNetConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
            Whether to freeze the batch normalization layers in the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Dimension of the layers in hybrid encoder.
        encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
            Multi level features input for encoder.
        feat_strides (`List[int]`, *optional*, defaults to `[8, 16, 32]`):
            Strides used in each feature map.
        encoder_layers (`int`, *optional*, defaults to 1):
            Total of layers to be used by the encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        encode_proj_layers (`List[int]`, *optional*, defaults to `[2]`):
            Indexes of the projected layers to be used in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The temperature parameter used to create the positional encodings.
        encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the general layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to computes the effective height and width of the position embeddings after taking
            into account the stride.
        normalize_before (`bool`, *optional*, defaults to `False`):
            Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
            feed-forward modules.
        hidden_expansion (`float`, *optional*, defaults to 1.0):
            Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers exclude hybrid encoder.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries.
        decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
            Multi level features dimension for decoder
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of input feature levels.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_denoising (`int`, *optional*, defaults to 100):
            The total number of denoising tasks or queries to be used for contrastive denoising.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The fraction of denoising labels to which random noise should be added.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale or magnitude of noise to be added to the bounding boxes.
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Indicates whether the initial query embeddings for the decoder should be learned during training
        anchor_image_size (`Tuple[int, int]`, *optional*):
            Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Whether to disable custom kernels.
        with_box_refine (`bool`, *optional*, defaults to `True`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the architecture has an encoder decoder structure.
        matcher_alpha (`float`, *optional*, defaults to 0.25):
            Parameter alpha used by the Hungarian Matcher.
        matcher_gamma (`float`, *optional*, defaults to 2.0):
            Parameter gamma used by the Hungarian Matcher.
        matcher_class_cost (`float`, *optional*, defaults to 2.0):
            The relative weight of the class loss used by the Hungarian Matcher.
        matcher_bbox_cost (`float`, *optional*, defaults to 5.0):
            The relative weight of the bounding box loss used by the Hungarian Matcher.
        matcher_giou_cost (`float`, *optional*, defaults to 2.0):
            The relative weight of the giou loss of used by the Hungarian Matcher.
        use_focal_loss (`bool`, *optional*, defaults to `True`):
            Parameter informing if focal focal should be used.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        focal_loss_alpha (`float`, *optional*, defaults to 0.75):
            Parameter alpha used to compute the focal loss.
        focal_loss_gamma (`float`, *optional*, defaults to 2.0):
            Parameter gamma used to compute the focal loss.
        weight_loss_vfl (`float`, *optional*, defaults to 1.0):
            Relative weight of the varifocal loss in the object detection loss.
        weight_loss_bbox (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 bounding box loss in the object detection loss.
        weight_loss_giou (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.0001):
            Relative classification weight of the 'no-object' class in the object detection loss.
        decoder_n_levels (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        decoder_offset_scale (`<fill_type>`, *optional*, defaults to 0.5): <fill_docstring>

    Examples:

    ```python
    >>> from transformers import RtDetrV2Config, RtDetrV2Model

    >>> # Initializing a RT-DETR configuration
    >>> configuration = RtDetrV2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RtDetrV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    decoder_n_levels (`int`, *optional*, defaults to 3): The number of feature levels used by the decoder.
    decoder_offset_scale (`float`, *optional*, defaults to 0.5): Scaling factor applied to the attention offsets in the decoder.
    """

    model_type = "rt_detr_v2"
    layer_types = ["basic", "bottleneck"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        initializer_range=0.01,
        initializer_bias_prior_prob=None,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        freeze_backbone_batch_norms=True,
        backbone_kwargs=None,
        # encoder HybridEncoder
        encoder_hidden_dim=256,
        encoder_in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        hidden_expansion=1.0,
        # decoder RtDetrV2Transformer
        d_model=256,
        num_queries=300,
        decoder_in_channels=[256, 256, 256],
        decoder_ffn_dim=1024,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=6,
        decoder_attention_heads=8,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        disable_custom_kernels=True,
        with_box_refine=True,
        is_encoder_decoder=True,
        # Loss
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        auxiliary_loss=True,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        eos_coefficient=1e-4,
        decoder_n_levels=3,  # default value
        decoder_offset_scale=0.5,  # default value
        **kwargs,
    ):
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.initializer_range = initializer_range
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `RtDetrV2-ResNet` backbone."
            )
            backbone_config = RtDetrV2ResNetConfig(
                num_channels=3,
                embedding_size=64,
                hidden_sizes=[256, 512, 1024, 2048],
                depths=[3, 4, 6, 3],
                layer_type="bottleneck",
                hidden_act="relu",
                downsample_in_first_stage=False,
                downsample_in_bottleneck=False,
                out_features=None,
                out_indices=[2, 3, 4],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.freeze_backbone_batch_norms = freeze_backbone_batch_norms
        self.backbone_kwargs = backbone_kwargs
        # encoder
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = encode_proj_layers
        self.encoder_layers = encoder_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.hidden_expansion = hidden_expansion
        self.num_queries = num_queries
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_in_channels = decoder_in_channels
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.auxiliary_loss = auxiliary_loss
        self.disable_custom_kernels = disable_custom_kernels
        self.with_box_refine = with_box_refine
        # Loss
        self.matcher_alpha = matcher_alpha
        self.matcher_gamma = matcher_gamma
        self.matcher_class_cost = matcher_class_cost
        self.matcher_bbox_cost = matcher_bbox_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.weight_loss_vfl = weight_loss_vfl
        self.weight_loss_bbox = weight_loss_bbox
        self.weight_loss_giou = weight_loss_giou
        self.eos_coefficient = eos_coefficient

        if not hasattr(self, "d_model"):
            self.d_model = d_model

        if not hasattr(self, "encoder_attention_heads"):
            self.encoder_attention_heads = encoder_attention_heads
        # add the new attributes with the given values or defaults
        self.decoder_n_levels = decoder_n_levels
        self.decoder_offset_scale = decoder_offset_scale

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

    @classmethod
    def from_backbone_configs(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`RtDetrV2Config`] (or a derived class) from a pre-trained backbone model configuration and DETR model
        configuration.

            Args:
                backbone_config ([`PretrainedConfig`]):
                    The backbone configuration.

            Returns:
                [`RtDetrV2Config`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )


__all__ = ["RtDetrV2Config", "RtDetrV2ResNetConfig"]
