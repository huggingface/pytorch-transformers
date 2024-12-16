# coding=utf-8
# Copyright 2021 Microsoft Research and the HuggingFace Inc. team.
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


from typing import Callable, List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxSequenceClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig


@flax.struct.dataclass
class FlaxBeitModelOutputWithPooling(FlaxBaseModelOutputWithPooling):
    """
    Class for outputs of [`FlaxBeitModel`].

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """


BEIT_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`BeitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def relative_position_index_init(window_size: Tuple[int, int]) -> jnp.ndarray:
    """
    get pair-wise relative position index for each token inside the window
    """
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = np.reshape(coords, (2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    relative_position_index = np.zeros(shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return jnp.array(relative_position_index)


def ones_with_scale(key, shape, scale, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * scale


class FlaxBeitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    rate: float

    @nn.module.compact
    def __call__(self, inputs, deterministic: Optional[bool] = True):
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            rng = self.make_rng("droppath")
            random_tensor = keep_prob + jax.random.uniform(rng, shape=shape, dtype=inputs.dtype)
            binary_tensor = jnp.floor(random_tensor)
            output = inputs / keep_prob * binary_tensor
            return output


class FlaxBeitPatchEmbeddings(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.num_channels = self.config.num_channels
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        patch_shape = (image_size // patch_size, image_size // patch_size)
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxBeitEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
        if self.config.use_mask_token:
            self.mask_token = self.param("mask_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
        self.patch_embeddings = FlaxBeitPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.param(
                "position_embeddings", nn.initializers.zeros, (1, num_patches + 1, self.config.hidden_size)
            )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, pixel_values, bool_masked_pos=None, deterministic=True):
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.shape

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        cls_tokens = cls_tokens.astype(embeddings.dtype)

        if bool_masked_pos is not None:
            mask_tokens = jnp.broadcast_to(self.mask_token, (batch_size, seq_len, self.config.hidden_size))
            mask_tokens = mask_tokens.astype(embeddings.dtype)
            # replace the masked visual tokens by mask_tokens
            w = jnp.expand_dims(bool_masked_pos, axis=-1)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        if self.config.use_absolute_position_embeddings:
            embeddings = embeddings + self.position_embeddings.astype(embeddings.dtype)

        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


class FlaxBeitRelativePositionBias(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
        self.relative_position_bias_table = self.param(
            "relative_position_bias_table",
            nn.initializers.zeros,
            (num_relative_distance, self.config.num_attention_heads),
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        self.relative_position_index = relative_position_index_init(self.window_size)

    def __call__(self):
        index = self.relative_position_index.reshape(-1)
        shape = (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        relative_position_bias = self.relative_position_bias_table[index].reshape(shape)  # Wh*Ww,Wh*Ww,nH
        return jnp.transpose(relative_position_bias, (2, 0, 1))


class FlaxBeitSelfAttention(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0 and not hasattr(
            self.config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {self.config.hidden_size,} is not a multiple of the number of attention "
                f"heads {self.config.num_attention_heads}."
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.relative_position_bias = (
            FlaxBeitRelativePositionBias(self.config, window_size=self.window_size, dtype=self.dtype)
            if self.window_size
            else None
        )

    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic: bool = True, output_attentions: bool = False
    ):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attention_bias = jnp.array(0.0, dtype=self.dtype)
        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            attention_bias = jnp.expand_dims(self.relative_position_bias(), 0)
            attention_bias = attention_bias.astype(query_states.dtype)

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_bias = attention_bias + relative_position_bias.astype(attention_bias.dtype)

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxBeitSelfOutput(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxBeitAttention(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxBeitSelfAttention(self.config, self.window_size, dtype=self.dtype)
        self.output = FlaxBeitSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic=True, output_attentions: bool = False
    ):
        attn_outputs = self.attention(
            hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        attn_output = self.output(attn_output, deterministic=deterministic)

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxBeitIntermediate(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)

        return hidden_states


class FlaxBeitOutput(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        return hidden_states


class FlaxBeitLayer(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    drop_path_rate: float
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxBeitAttention(self.config, self.window_size, dtype=self.dtype)
        self.intermediate = FlaxBeitIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBeitOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.drop_path = FlaxBeitDropPath(rate=self.drop_path_rate)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        self.init_values = self.config.layer_scale_init_value
        if self.init_values > 0:
            self.lambda_1 = self.param("lambda_1", ones_with_scale, (self.config.hidden_size), self.init_values)
            self.lambda_2 = self.param("lambda_2", ones_with_scale, (self.config.hidden_size), self.init_values)
        else:
            self.lambda_1 = None
            self.lambda_2 = None

    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic: bool = True, output_attentions: bool = False
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in BEiT, layernorm is applied before self-attention
            relative_position_bias,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1.astype(attention_output.dtype) * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output, deterministic=deterministic) + hidden_states

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, deterministic=deterministic)

        # apply lambda_2 if present
        if self.lambda_2 is not None:
            layer_output = self.lambda_2.astype(layer_output.dtype) * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output, deterministic=deterministic) + hidden_states

        outputs = (layer_output,)

        if output_attentions:
            outputs += (self_attention_outputs[1],)

        return outputs


class FlaxBeitLayerCollection(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    drop_path_rates: List[float]
    relative_position_bias: Callable[[], jnp.ndarray]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxBeitLayer(
                self.config,
                window_size=self.window_size if self.config.use_relative_position_bias else None,
                drop_path_rate=self.drop_path_rates[i],
                name=str(i),
                dtype=self.dtype,
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            relative_position_bias = self.relative_position_bias() if self.relative_position_bias is not None else None
            layer_outputs = layer(
                hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxBeitEncoder(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.use_shared_relative_position_bias:
            self.relative_position_bias = FlaxBeitRelativePositionBias(
                config=self.config, window_size=self.window_size, dtype=self.dtype
            )

        # stochastic depth decay rule
        drop_path_rates = list(np.linspace(0, self.config.drop_path_rate, self.config.num_hidden_layers))
        self.layer = FlaxBeitLayerCollection(
            self.config,
            window_size=self.window_size,
            drop_path_rates=drop_path_rates,
            relative_position_bias=self.relative_position_bias
            if self.config.use_shared_relative_position_bias
            else None,
            dtype=self.dtype,
        )

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxBeitPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BeitConfig
    base_model_prefix = "beit"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: BeitConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        dropout_rng, droppath_rng = jax.random.split(dropout_rng)
        rngs = {"params": params_rng, "dropout": dropout_rng, "droppath": droppath_rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
        bool_masked_pos=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            dropout_rng, droppath_rng = jax.random.split(dropout_rng)
            rngs["dropout"] = dropout_rng
            rngs["droppath"] = droppath_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            bool_masked_pos,
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxBeitPooler(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.use_mean_pooling:
            self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        if self.config.use_mean_pooling:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(jnp.mean(patch_tokens, axis=1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output


class FlaxBeitModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBeitEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBeitEncoder(
            self.config, window_size=self.embeddings.patch_embeddings.patch_shape, dtype=self.dtype
        )
        if not self.config.use_mean_pooling:
            self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxBeitPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, bool_masked_pos, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if not self.config.use_mean_pooling:
            hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBeitModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare Beit Model transformer outputting raw hidden-states without any specific head on top.",
    BEIT_START_DOCSTRING,
)
class FlaxBeitModel(FlaxBeitPreTrainedModel):
    module_class = FlaxBeitModule


FLAX_BEIT_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxBeitModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxBeitModel, FLAX_BEIT_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxBeitModel, output_type=FlaxBeitModelOutputWithPooling, config_class=BeitConfig)


class FlaxBeitForMaskedImageModelingModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.beit = FlaxBeitModule(self.config, add_pooling_layer=False, dtype=self.dtype)

        # Classifier head
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return output

        return FlaxMaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "Beit Model transformer with a 'language' modeling head on top (to predict visual tokens).",
    BEIT_START_DOCSTRING,
)
class FlaxBeitForMaskedImageModeling(FlaxBeitPreTrainedModel):
    module_class = FlaxBeitForMaskedImageModelingModule


FLAX_BEIT_MLM_DOCSTRING = """
    bool_masked_pos (`numpy.ndarray` of shape `(batch_size, num_patches)`):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

overwrite_call_docstring(FlaxBeitForMaskedImageModeling, FLAX_BEIT_MLM_DOCSTRING)
append_replace_return_docstrings(
    FlaxBeitForMaskedImageModeling, output_type=FlaxMaskedLMOutput, config_class=BeitConfig
)


class FlaxBeitForImageClassificationModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.beit = FlaxBeitModule(config=self.config, dtype=self.dtype, add_pooling_layer=True)
        self.classifier = nn.Dense(
            self.config.num_labels,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
class FlaxBeitForImageClassification(FlaxBeitPreTrainedModel):
    module_class = FlaxBeitForImageClassificationModule


FLAX_BEIT_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxBeitForImageClassification
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    >>> model = FlaxBeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = logits.argmax(-1).item()
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
    ```
"""

overwrite_call_docstring(FlaxBeitForImageClassification, FLAX_BEIT_CLASSIF_DOCSTRING)
append_replace_return_docstrings(
    FlaxBeitForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=BeitConfig
)


__all__ = [
    "FlaxBeitForImageClassification",
    "FlaxBeitForMaskedImageModeling",
    "FlaxBeitModel",
    "FlaxBeitPreTrainedModel",
]
