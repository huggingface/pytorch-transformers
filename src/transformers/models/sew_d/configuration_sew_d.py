# coding=utf-8
# Copyright 2021 ASAPP Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""SEW-D model configuration"""

import functools
import operator

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class SEWDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SEWDModel`]. It is used to instantiate a SEW-D
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SEW-D
    [asapp/sew-d-tiny-100k](https://huggingface.co/asapp/sew-d-tiny-100k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32):
            Vocabulary size of the SEW-D model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SEWD`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        squeeze_factor (`int`, *optional*, defaults to 2):
            Sequence length downsampling factor after the encoder and upsampling factor after the transformer.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        position_buckets (`int`, *optional*, defaults to 256):
            The maximum size of relative position embeddings.
        share_att_key (`bool`, *optional*, defaults to `True`):
            Whether to share attention key with c2p and p2c.
        relative_attention (`bool`, *optional*, defaults to `True`):
            Whether to use relative position encoding.
        pos_att_type (`Tuple[str]`, *optional*, defaults to `("p2c", "c2p")`):
            The type of relative position attention, it can be a combination of `("p2c", "c2p")`, e.g. `("p2c")`,
            `("p2c", "c2p")`, `("p2c", "c2p")`.
        norm_rel_ebd (`str`, *optional*, defaults to `"layer_norm"`):
            Whether to use layer norm in relative embedding (`"layer_norm"` if yes)
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_python"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"gelu_python"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Deprecated. Not used by the model and will be removed in a future version.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        final_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the final projection layer of [`SEWDForCTC`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon used by the layer normalization layers in the transformer encoder.
        feature_layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization after the feature encoder.
        feat_extract_norm (`str`, *optional*, defaults to `"group"`):
            The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
            normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the feature encoder.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        conv_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
            of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
        conv_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
            length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
        diversity_loss_weight (`int`, *optional*, defaults to 0.1):
            The weight of the codebook diversity loss component.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`SEWDForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`SEWDForCTC`].
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`Wav2Vec2ForSequenceClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification.

    Example:

    ```python
    >>> from transformers import SEWDConfig, SEWDModel

    >>> # Initializing a SEW-D asapp/sew-d-tiny-100k style configuration
    >>> configuration = SEWDConfig()

    >>> # Initializing a model (with random weights) from the asapp/sew-d-tiny-100k style configuration
    >>> model = SEWDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sew-d"

    def __init__(
        self,
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        squeeze_factor=2,
        max_position_embeddings=512,
        position_buckets=256,
        share_att_key=True,
        relative_attention=True,
        pos_att_type=("p2c", "c2p"),
        norm_rel_ebd="layer_norm",
        hidden_act="gelu_python",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        feature_layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512),
        conv_stride=(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1),
        conv_kernel=(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.squeeze_factor = squeeze_factor
        self.max_position_embeddings = max_position_embeddings
        self.position_buckets = position_buckets
        self.share_att_key = share_att_key
        self.relative_attention = relative_attention
        self.norm_rel_ebd = norm_rel_ebd
        self.pos_att_type = list(pos_att_type)
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self._hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layer_norm_eps = layer_norm_eps
        self.feature_layer_norm_eps = feature_layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. "
                "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`, "
                f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride) "
                f"= {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # sequence classification
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.classifier_proj_size = classifier_proj_size

    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        output = super().to_dict()
        output["hidden_dropout"] = output.pop("_hidden_dropout")
        return output


__all__ = ["SEWDConfig"]
