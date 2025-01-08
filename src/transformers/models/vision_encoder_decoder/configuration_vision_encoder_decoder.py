# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


if TYPE_CHECKING:
    from ... import PreTrainedTokenizerBase, TensorType

logger = logging.get_logger(__name__)


class VisionEncoderDecoderConfig(PretrainedConfig):
    r"""
    [`VisionEncoderDecoderConfig`] is the configuration class to store the configuration of a
    [`VisionEncoderDecoderModel`]. It is used to instantiate a Vision-Encoder-Text-Decoder model according to the
    specified arguments, defining the encoder and decoder configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    >>> # Initializing a ViT & BERT style configuration
    >>> config_encoder = ViTConfig()
    >>> config_decoder = BertConfig()

    >>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a ViTBert model (with random weights) from a ViT & google-bert/bert-base-uncased style configurations
    >>> model = VisionEncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""

    model_type = "vision-encoder-decoder"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}"
            )

        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`VisionEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`VisionEncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


class VisionEncoderDecoderEncoderOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict({"last_hidden_state": {0: "batch", 1: "encoder_sequence"}})


class VisionEncoderDecoderDecoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict()
        common_inputs["input_ids"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        common_inputs["attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        common_inputs["encoder_hidden_states"] = {0: "batch", 1: "encoder_sequence"}

        return common_inputs

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        import torch

        common_inputs = OrderedDict()

        dummy_input = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        batch, encoder_sequence = dummy_input["input_ids"].shape
        encoder_hidden_states_shape = (batch, encoder_sequence, self._config.encoder_hidden_size)
        common_inputs["input_ids"] = dummy_input.pop("input_ids")
        common_inputs["attention_mask"] = dummy_input.pop("attention_mask")
        common_inputs["encoder_hidden_states"] = torch.zeros(encoder_hidden_states_shape)

        return common_inputs


class VisionEncoderDecoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> None:
        pass

    def get_encoder_config(self, encoder_config: PretrainedConfig) -> OnnxConfig:
        r"""
        Returns ONNX encoder config for `VisionEncoderDecoder` model.

        Args:
            encoder_config (`PretrainedConfig`):
                The encoder model's configuration to use when exporting to ONNX.

        Returns:
            [`VisionEncoderDecoderEncoderOnnxConfig`]: An instance of the ONNX configuration object
        """
        return VisionEncoderDecoderEncoderOnnxConfig(encoder_config)

    def get_decoder_config(
        self, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, feature: str = "default"
    ) -> OnnxConfig:
        r"""
        Returns ONNX decoder config for `VisionEncoderDecoder` model.

        Args:
            encoder_config (`PretrainedConfig`):
                The encoder model's configuration to use when exporting to ONNX.
            decoder_config (`PretrainedConfig`):
                The decoder model's configuration to use when exporting to ONNX
            feature (`str`, *optional*):
                The type of feature to export the model with.

        Returns:
            [`VisionEncoderDecoderDecoderOnnxConfig`]: An instance of the ONNX configuration object.
        """
        decoder_config.encoder_hidden_size = encoder_config.hidden_size
        return VisionEncoderDecoderDecoderOnnxConfig(decoder_config, feature)


__all__ = ["VisionEncoderDecoderConfig", "VisionEncoderDecoderOnnxConfig"]
