# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
"""Tensorflow mT5 model."""

from ...utils import logging
from ..t5.modeling_tf_t5 import TFT5EncoderModel, TFT5ForConditionalGeneration, TFT5Model
from .configuration_mt5 import MT5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"


class TFMT5Model(TFT5Model):
    r"""
    This class overrides [`TFT5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```python
    >>> from transformers import TFMT5Model, AutoTokenizer

    >>> model = TFMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="tf")
    >>> labels = tokenizer(text_target=summary, return_tensors="tf")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""

    model_type = "mt5"
    config_class = MT5Config


class TFMT5ForConditionalGeneration(TFT5ForConditionalGeneration):
    r"""
    This class overrides [`TFT5ForConditionalGeneration`]. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples:

    ```python
    >>> from transformers import TFMT5ForConditionalGeneration, AutoTokenizer

    >>> model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="tf")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    model_type = "mt5"
    config_class = MT5Config


class TFMT5EncoderModel(TFT5EncoderModel):
    r"""
    This class overrides [`TFT5EncoderModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.

    Examples:

    ```python
    >>> from transformers import TFMT5EncoderModel, AutoTokenizer

    >>> model = TFMT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="tf").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""

    model_type = "mt5"
    config_class = MT5Config


__all__ = ["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"]
