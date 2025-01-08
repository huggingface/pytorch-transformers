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
"""LayoutLMv3 model configuration"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import logging


if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType


logger = logging.get_logger(__name__)


class LayoutLMv3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutLMv3Model`]. It is used to instantiate an
    LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
    [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the LayoutLMv3 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`LayoutLMv3Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LayoutLMv3Model`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the 2D position embedding might ever be used with. Typically set this to something
            large just in case (e.g., 1024).
        coordinate_size (`int`, *optional*, defaults to `128`):
            Dimension of the coordinate embeddings.
        shape_size (`int`, *optional*, defaults to `128`):
            Dimension of the width and height embeddings.
        has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a relative attention bias in the self-attention mechanism.
        rel_pos_bins (`int`, *optional*, defaults to 32):
            The number of relative position bins to be used in the self-attention mechanism.
        max_rel_pos (`int`, *optional*, defaults to 128):
            The maximum number of relative positions to be used in the self-attention mechanism.
        max_rel_2d_pos (`int`, *optional*, defaults to 256):
            The maximum number of relative 2D positions in the self-attention mechanism.
        rel_2d_pos_bins (`int`, *optional*, defaults to 64):
            The number of 2D relative position bins in the self-attention mechanism.
        has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a spatial attention bias in the self-attention mechanism.
        visual_embed (`bool`, *optional*, defaults to `True`):
            Whether or not to add patch embeddings.
        input_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of the images.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of channels of the images.
        patch_size (`int`, *optional*, defaults to `16`)
            The size (resolution) of the patches.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Example:

    ```python
    >>> from transformers import LayoutLMv3Config, LayoutLMv3Model

    >>> # Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
    >>> configuration = LayoutLMv3Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
    >>> model = LayoutLMv3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "layoutlmv3"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        rel_pos_bins=32,
        max_rel_pos=128,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        has_spatial_attention_bias=True,
        text_embed=True,
        visual_embed=True,
        input_size=224,
        num_channels=3,
        patch_size=16,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.text_embed = text_embed
        self.visual_embed = visual_embed
        self.input_size = input_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.classifier_dropout = classifier_dropout


class LayoutLMv3OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.12")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # The order of inputs is different for question answering and sequence classification
        if self.task in ["question-answering", "sequence-classification"]:
            return OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "sequence"}),
                    ("attention_mask", {0: "batch", 1: "sequence"}),
                    ("bbox", {0: "batch", 1: "sequence"}),
                    ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ]
            )
        else:
            return OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "sequence"}),
                    ("bbox", {0: "batch", 1: "sequence"}),
                    ("attention_mask", {0: "batch", 1: "sequence"}),
                    ("pixel_values", {0: "batch", 1: "num_channels"}),
                ]
            )

    @property
    def atol_for_validation(self) -> float:
        return 1e-5

    @property
    def default_onnx_opset(self) -> int:
        return 12

    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
    ) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            processor ([`ProcessorMixin`]):
                The processor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2).
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the processor will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Any]: holding the kwargs to provide to the model's forward function
        """

        # A dummy image is used so OCR should not be applied
        setattr(processor.image_processor, "apply_ocr", False)

        # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )
        # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
        token_to_add = processor.tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )
        # Generate dummy inputs according to compute batch and sequence
        dummy_text = [[" ".join([processor.tokenizer.unk_token]) * seq_length]] * batch_size

        # Generate dummy bounding boxes
        dummy_bboxes = [[[48, 84, 73, 128]]] * batch_size

        # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
        # batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
        dummy_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)

        inputs = dict(
            processor(
                dummy_image,
                text=dummy_text,
                boxes=dummy_bboxes,
                return_tensors=framework,
            )
        )

        return inputs


__all__ = ["LayoutLMv3Config", "LayoutLMv3OnnxConfig"]
