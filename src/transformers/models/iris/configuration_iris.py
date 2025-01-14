# coding=utf-8
# Copyright 2024 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""IRIS model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class IrisConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`IrisModel`]. It is used to
    instantiate a IRIS model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
    Iris [ruffy369/iris-breakout](https://huggingface.co/ruffy369/iris-breakout) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_actions (`int`, *optional*, defaults to 4):
            The number of actions or size of the output action space for the Atari environment
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of Discrete Autoencoder and World Model
        device (`str`, *optional*, defaults to `"cuda:0"`):
            The device on which the model is (assuming that all the module parameters are on the samedevice)
        embed_dim_discrete_autoencoder (`int`, *optional*, defaults to 512):
            The embedding dim of Discrete Autoencoder
        resolution (`int`, *optional*, defaults to 64):
            The resolution of image frame observation passed to encoder
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels of the input image
        z_channels (`int`, *optional*, defaults to 512):
            Number of channels for tokens encoded by Encoder which in turn will be the input for Decoder
        ch (`int`, *optional*, defaults to 64):
            For calculation of number of in channels and out channels in ResnetBlock for Discrete Autoencoder
        ch_mult (`list`, *optional*, defaults to `[1, 1, 1, 1, 1]`):
            len of it is equal to number of resolutions in Encoder and Decoder of Discrete Autoencoder and is used for calculation of number of in channels and out channels in ResnetBlock for Discrete Autoencoder.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of Resnet Blocks in Encoder and Decoder of Discrete Autoencoder.
        attn_resolutions (`list`, *optional*, defaults to `[8, 16]`):
            The resolutions at which an attention layer is used for Encoder and Decoder of Discrete Autoencoder.
        out_ch (`int`, *optional*, defaults to 3):
            Number of out channels for Decoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for ResnetBlocks in Encoder and Decoder.
        use_original_obs_actor_critic (`bool`, *optional*, defaults to `False`):
            Tells wheter the actor critic should use original observation or reconstructed image from Decoder for predicting action
        tokens_per_block (`int`, *optional*, defaults to 17):
            Number of tokens per block in World Model's transformer
        max_blocks (`int`, *optional*, defaults to 20):
            Max number of blocks in World Model's transformer
        attention (`string`, *optional*, defaults to `"causal"`):
            Type of attention in World Model's transformer self attention
        num_layers (`int`, *optional*, defaults to 10):
            Number of layers in World Model's transformer
        num_heads (`int`, *optional*, defaults to 4):
            Number of attention heads in World Model's transformer
        embed_dim_world_model (`int`, *optional*, defaults to 256):
            The embedding dim of World Model
        embed_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for sequences fed to Transformer
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for attention layers applied to last state in World Model's Transformer
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for attention weights softmax layers in World Model's Transformer
        grad_acc_steps_discrete_autoencoder (`int`, *optional*, defaults to 1):
            Number of Gradient steps for Discrete Autoencoder
        grad_acc_steps_world_model (`int`, *optional*, defaults to 1):
            Number of Gradient steps for World Model
        grad_acc_steps_actor_critic (`int`, *optional*, defaults to 1):
            Number of Gradient steps for Actor Critic
        gamma (`float`, *optional*, defaults to 0.995):
            Discount Factor for Actor Critic
        lambda_ (`float`, *optional*, defaults to 0.95):
            Lambda for lambda returns in Actor Critic
        entropy_weight (`float`, *optional*, defaults to 0.001):
            Entropy weight for entropy loss in Actor Critic
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.


    Example:

    ```python
    >>> from transformers import IrisConfig, IrisModel

    >>> # Initializing a Iris configuration
    >>> configuration = IrisConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = IrisModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "iris"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "embed_dim_world_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_heads",
    }

    def __init__(
        self,
        num_actions=4,
        vocab_size=512,
        device="cuda:0",
        embed_dim_discrete_autoencoder=512,
        resolution=64,
        in_channels=3,
        z_channels=512,
        ch=64,
        ch_mult=[1, 1, 1, 1, 1],
        num_res_blocks=2,
        attn_resolutions=[8, 16],
        out_ch=3,
        dropout=0.0,
        use_original_obs_actor_critic=False,
        tokens_per_block=17,
        max_blocks=20,
        attention="causal",
        num_layers=10,
        num_heads=4,
        embed_dim_world_model=256,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        grad_acc_steps_discrete_autoencoder=1,
        grad_acc_steps_world_model=1,
        grad_acc_steps_actor_critic=1,
        gamma=0.995,
        lambda_=0.95,
        entropy_weight=0.001,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.num_actions = num_actions
        self.vocab_size = vocab_size
        self.device = device
        self.embed_dim_discrete_autoencoder = embed_dim_discrete_autoencoder
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.out_ch = out_ch
        self.dropout = dropout
        self.use_original_obs_actor_critic = use_original_obs_actor_critic
        self.tokens_per_block = tokens_per_block
        self.max_blocks = max_blocks
        self.max_tokens = self.tokens_per_block * self.max_blocks
        self.attention = attention
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim_world_model = embed_dim_world_model
        self.embed_pdrop = embed_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.sequence_length = self.max_blocks
        self.grad_acc_steps_discrete_autoencoder = grad_acc_steps_discrete_autoencoder
        self.grad_acc_steps_world_model = grad_acc_steps_world_model
        self.grad_acc_steps_actor_critic = grad_acc_steps_actor_critic
        self.imagine_horizon_train_actor_critic = self.sequence_length
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_weight = entropy_weight
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
