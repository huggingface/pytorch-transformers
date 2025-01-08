# coding=utf-8
# Copyright 2024 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Wav2Vec2Bert model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Wav2Vec2BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Wav2Vec2BertModel`]. It is used to
    instantiate an Wav2Vec2Bert model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2Bert
    [facebook/wav2vec2-bert-rel-pos-large](https://huggingface.co/facebook/wav2vec2-bert-rel-pos-large)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the Wav2Vec2Bert model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`Wav2Vec2BertModel`]. Vocabulary size of the
            model. Defines the different tokens that can be represented by the *inputs_ids* passed to the forward
            method of [`Wav2Vec2BertModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        feature_projection_input_dim (`int`, *optional*, defaults to 160):
            Input dimension of this model, i.e the dimension after processing input audios with [`SeamlessM4TFeatureExtractor`] or [`Wav2Vec2BertProcessor`].
        hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the feature projection.
        final_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the final projection layer of [`Wav2Vec2BertForCTC`].
        layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more
            details.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates `mask_time_prob*len(time_axis)/mask_time_length ``independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2):
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if `mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks`.
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0):
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`Wav2Vec2BertForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`Wav2Vec2BertForCTC`].
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`Wav2Vec2BertForSequenceClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 768):
            Dimensionality of the projection before token mean-pooling for classification.
        tdnn_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 1500)`):
            A tuple of integers defining the number of output channels of each 1D convolutional layer in the *TDNN*
            module of the *XVector* model. The length of *tdnn_dim* defines the number of *TDNN* layers.
        tdnn_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 3, 3, 1, 1)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the *TDNN* module of the
            *XVector* model. The length of *tdnn_kernel* has to match the length of *tdnn_dim*.
        tdnn_dilation (`Tuple[int]` or `List[int]`, *optional*, defaults to `(1, 2, 3, 1, 1)`):
            A tuple of integers defining the dilation factor of each 1D convolutional layer in *TDNN* module of the
            *XVector* model. The length of *tdnn_dilation* has to match the length of *tdnn_dim*.
        xvector_output_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the *XVector* embedding vectors.
        pad_token_id (`int`, *optional*, defaults to 0): The id of the _beginning-of-stream_ token.
        bos_token_id (`int`, *optional*, defaults to 1): The id of the _padding_ token.
        eos_token_id (`int`, *optional*, defaults to 2): The id of the _end-of-stream_ token.
        add_adapter (`bool`, *optional*, defaults to `False`):
            Whether a convolutional attention network should be stacked on top of the Wav2Vec2Bert Encoder. Can be very
            useful for warm-starting Wav2Vec2Bert for SpeechEncoderDecoder models.
        adapter_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adapter_stride (`int`, *optional*, defaults to 2):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        num_adapter_layers (`int`, *optional*, defaults to 1):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        adapter_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the adapter layers. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
        use_intermediate_ffn_before_adapter (`bool`, *optional*, defaults to `False`):
            Whether an intermediate feed-forward block should be stacked on top of the Wav2Vec2Bert Encoder and before the adapter network.
             Only relevant if `add_adapter is True`.
        output_hidden_size (`int`, *optional*):
            Dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
            if `add_adapter is True`.
        position_embeddings_type (`str`, *optional*, defaults to `"relative_key"`):
            Can be specified to :
                - `rotary`, for rotary position embeddings.
                - `relative`, for relative position embeddings.
                - `relative_key`, for relative position embeddings as defined by Shaw in [Self-Attention
            with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            If left to `None`, no relative position embeddings is applied.
        rotary_embedding_base (`int`, *optional*, defaults to 10000):
            If `"rotary"` position embeddings are used, defines the size of the embedding base.
        max_source_positions (`int`, *optional*, defaults to 5000):
            if `"relative"` position embeddings are used, defines the maximum source input positions.
        left_max_position_embeddings (`int`, *optional*, defaults to 64):
            If `"relative_key"` (aka Shaw) position embeddings are used, defines the left clipping value for relative positions.
        right_max_position_embeddings (`int`, *optional*, defaults to 8):
            If `"relative_key"` (aka Shaw) position embeddings are used, defines the right clipping value for relative positions.
        conv_depthwise_kernel_size (`int`, *optional*, defaults to 31):
            Kernel size of convolutional depthwise 1D layer in Conformer blocks.
        conformer_conv_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all convolutional layers in Conformer blocks.
    Example:

    ```python
    >>> from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    >>> # Initializing a Wav2Vec2Bert facebook/wav2vec2-bert-rel-pos-large style configuration
    >>> configuration = Wav2Vec2BertConfig()

    >>> # Initializing a model (with random weights) from the facebook/wav2vec2-bert-rel-pos-large style configuration
    >>> model = Wav2Vec2BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "wav2vec2-bert"

    def __init__(
        self,
        vocab_size=None,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        feature_projection_input_dim=160,
        hidden_act="swish",
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=768,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        xvector_output_dim=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        add_adapter=False,
        adapter_kernel_size=3,
        adapter_stride=2,
        num_adapter_layers=1,
        adapter_act="relu",
        use_intermediate_ffn_before_adapter=False,
        output_hidden_size=None,
        position_embeddings_type="relative_key",
        rotary_embedding_base=10000,
        max_source_positions=5000,
        left_max_position_embeddings=64,
        right_max_position_embeddings=8,
        conv_depthwise_kernel_size=31,
        conformer_conv_dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.feature_projection_input_dim = feature_projection_input_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.max_source_positions = max_source_positions

        if position_embeddings_type is not None and position_embeddings_type not in [
            "rotary",
            "relative",
            "relative_key",
        ]:
            raise ValueError(
                """
                `position_embeddings_type` is not valid. It must be one of the following values:
                `["rotary", "relative", "relative_key"]` or left as `None`.
                """
            )
        self.position_embeddings_type = position_embeddings_type
        self.rotary_embedding_base = rotary_embedding_base
        self.left_max_position_embeddings = left_max_position_embeddings
        self.right_max_position_embeddings = right_max_position_embeddings

        # Conformer-block related
        self.conv_depthwise_kernel_size = conv_depthwise_kernel_size
        self.conformer_conv_dropout = conformer_conv_dropout

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

        # adapter
        self.add_adapter = add_adapter
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_stride = adapter_stride
        self.num_adapter_layers = num_adapter_layers
        self.adapter_act = adapter_act
        self.output_hidden_size = output_hidden_size if output_hidden_size is not None else hidden_size
        if use_intermediate_ffn_before_adapter and not add_adapter:
            raise ValueError("`use_intermediate_ffn_before_adapter` is `True` but `add_adapter` is `False`.")
        self.use_intermediate_ffn_before_adapter = use_intermediate_ffn_before_adapter

        # SequenceClassification-specific parameter. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size

        # XVector-specific parameters. Feel free to ignore for other classes.
        self.tdnn_dim = list(tdnn_dim)
        self.tdnn_kernel = list(tdnn_kernel)
        self.tdnn_dilation = list(tdnn_dilation)
        self.xvector_output_dim = xvector_output_dim

    @property
    def inputs_to_logits_ratio(self):
        ratio = self.feature_projection_input_dim * 2
        if self.add_adapter:
            ratio = ratio * (self.adapter_stride**self.num_adapter_layers)
        return ratio


__all__ = ["Wav2Vec2BertConfig"]
