# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""Flax DINOv2 model."""

import collections.abc
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_dinov2 import Dinov2Config


DINOV2_START_DOCSTRING = r"""

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
        config ([`Dinov2Config`]): Model configuration class with all the parameters of the model.
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

DINOV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`Dinov2ImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxDinov2PatchEmbeddings(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    # Copied from transformers.models.vit.modeling_flax_vit.FlaxViTPatchEmbeddings.__call__
    def __call__(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxDinov2Embeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        self.mask_token = self.param(
            "mask_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, self.config.hidden_size),
        )
        self.patch_embeddings = FlaxDinov2PatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 1, self.config.hidden_size),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, config, hidden_states, height, width, position_embeddings):
        num_patches = hidden_states.shape[1] - 1
        num_positions = position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return position_embeddings
        class_pos_embed = position_embeddings[:, 0]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = hidden_states.shape[-1]

        h = height // config.patch_size
        w = width // config.patch_size
        height, width = h + 0.1, w + 0.1

        patch_pos_embed = patch_pos_embed.reshape(
            (1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        )
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))
        target_dtype = patch_pos_embed.dtype
        new_height_ratio = jnp.float32(height / math.sqrt(num_positions))
        new_width_ratio = jnp.float32(width / math.sqrt(num_positions))

        scale = jnp.array([new_height_ratio, new_width_ratio], dtype=jnp.float32)
        translation = jnp.array([0.0, 0.0], dtype=jnp.float32)

        patch_pos_embed = jax.image.scale_and_translate(
            patch_pos_embed.astype(jnp.float32),
            shape=(patch_pos_embed.shape[0], patch_pos_embed.shape[1], h, w),
            spatial_dims=(2, 3),
            scale=scale,
            translation=translation,
            method="bicubic",
            antialias=False,
        )
        patch_pos_embed = patch_pos_embed.astype(target_dtype)
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 2, 3, 1)).reshape((hidden_states.shape[0], -1, dim))

        return jnp.concatenate((class_pos_embed[jnp.newaxis, :], patch_pos_embed), axis=1)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.dtype
        height, width = pixel_values.shape[1], pixel_values.shape[2]

        embeddings = self.patch_embeddings(pixel_values.astype(target_dtype))

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        embeddings = embeddings + self.interpolate_pos_encoding(
            self.config, embeddings, height, width, self.position_embeddings
        )

        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfAttention with ViT->Dinov2
class FlaxDinov2SelfAttention(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
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

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
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


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfOutput with ViT->Dinov2
class FlaxDinov2SelfOutput(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTAttention with ViT->Dinov2
class FlaxDinov2Attention(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxDinov2SelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxDinov2SelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


def ones_with_scale(key, shape, scale, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * scale


class FlaxDinov2LayerScale(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.lambda1 = self.config.layerscale_value * self.param(
            "lambda1",
            jax.nn.initializers.ones,
            (self.config.hidden_size,),
        )
        self.lambda1 = self.lambda1 * self.config.layerscale_value

    def __call__(self, hidden_states):
        return self.lambda1 * hidden_states


# Copied from transformers.models.beit.modeling_flax_beit.FlaxBeitDropPath with Beit -> Dinov2
class FlaxDinov2DropPath(nn.Module):
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


class FlaxDinov2MLP(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.fc1 = nn.Dense(
            self.config.hidden_size * self.config.mlp_ratio,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.fc2 = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        if isinstance(self.config.hidden_act, str):
            self.act = ACT2FN[self.config.hidden_act]
        else:
            self.act = self.config.hidden_act

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxDinov2SwiGLUFFN(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        hidden_features = int(self.config.hidden_size * self.config.mlp_ratio)
        hidden_features = (int(self.hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Dense(
            2 * hidden_features,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.weights_out = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        hidden_states = self.weights_in(hidden_states)
        x1, x2 = jnp.split(hidden_states, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        return self.weights_out(hidden)


class FlaxDinov2Layer(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.attention = FlaxDinov2Attention(self.config, dtype=self.dtype)
        self.layer_scale1 = FlaxDinov2LayerScale(self.config, dtype=self.dtype)
        self.drop_path = FlaxDinov2DropPath(self.config.drop_path_rate)
        self.norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        if self.config.use_swiglu_ffn:
            self.mlp = FlaxDinov2SwiGLUFFN(self.config, dtype=self.dtype)
        else:
            self.mlp = FlaxDinov2MLP(self.config, dtype=self.dtype)

        self.layer_scale2 = FlaxDinov2LayerScale(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)

        outputs = self_attention_outputs[1:]

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTLayerCollection with ViT->Dinov2
class FlaxDinov2LayerCollection(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxDinov2Layer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
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

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

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


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTEncoder with ViT->Dinov2
class FlaxDinov2Encoder(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxDinov2LayerCollection(self.config, dtype=self.dtype)

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


class FlaxDinov2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Dinov2Config
    base_model_prefix = "dinov2"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Dinov2Config,
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

    @add_start_docstrings_to_model_forward(DINOV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
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
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxDinov2Module(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxDinov2Embeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxDinov2Encoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        encoder_outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The bare Dinov2 Model transformer outputting raw hidden-states without any specific head on top.",
    DINOV2_START_DOCSTRING,
)
class FlaxDinov2Model(FlaxDinov2PreTrainedModel):
    module_class = FlaxDinov2Module


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxDinov2Model
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    >>> model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxDinov2Model, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(
    FlaxDinov2Model, output_type=FlaxBaseModelOutputWithPooling, config_class=Dinov2Config
)


class FlaxDinov2ForImageClassificationModule(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dinov2 = FlaxDinov2Module(config=self.config, dtype=self.dtype)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.dinov2(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        cls_token = hidden_states[:, 0]
        patch_tokens = hidden_states[:, 1:]
        linear_input = jnp.concatenate([cls_token, patch_tokens.mean(axis=1)], axis=-1)

        logits = self.classifier(linear_input)

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
    Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    DINOV2_START_DOCSTRING,
)
class FlaxDinov2ForImageClassification(FlaxDinov2PreTrainedModel):
    module_class = FlaxDinov2ForImageClassificationModule


FLAX_VISION_CLASSIFICATION_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxDinov2ForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    >>> model = FlaxDinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
    ```
"""

overwrite_call_docstring(FlaxDinov2ForImageClassification, FLAX_VISION_CLASSIFICATION_DOCSTRING)
append_replace_return_docstrings(
    FlaxDinov2ForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=Dinov2Config
)


__all__ = ["FlaxDinov2ForImageClassification", "FlaxDinov2Model", "FlaxDinov2PreTrainedModel"]
