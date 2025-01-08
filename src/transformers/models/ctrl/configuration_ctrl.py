# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Salesforce CTRL configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class CTRLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CTRLModel`] or a [`TFCTRLModel`]. It is used to
    instantiate a CTRL model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Salesforce/ctrl](https://huggingface.co/Salesforce/ctrl) architecture from SalesForce.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 246534):
            Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CTRLModel`] or [`TFCTRLModel`].
        n_positions (`int`, *optional*, defaults to 256):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 1280):
            Dimensionality of the embeddings and hidden states.
        dff (`int`, *optional*, defaults to 8192):
            Dimensionality of the inner dimension of the feed forward networks (FFN).
        n_layer (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-06):
            The epsilon to use in the layer normalization layers
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).


    Examples:

    ```python
    >>> from transformers import CTRLConfig, CTRLModel

    >>> # Initializing a CTRL configuration
    >>> configuration = CTRLConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CTRLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ctrl"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=246534,
        n_positions=256,
        n_embd=1280,
        dff=8192,
        n_layer=48,
        n_head=16,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dff = dff
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.use_cache = use_cache

        super().__init__(**kwargs)


__all__ = ["CTRLConfig"]
