# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
"""TF 2.0 ViT MAE (masked autoencoder) model."""

from __future__ import annotations

import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ViTMAEConfig"
_CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"


@dataclass
class TFViTMAEModelOutput(ModelOutput):
    """
    Class for TFViTMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFViTMAEDecoderOutput(ModelOutput):
    """
    Class for TFViTMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFViTMAEForPreTrainingOutput(ModelOutput):
    """
    Class for TFViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`tf.Tensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`tf.Tensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the position
        embeddings (with or without classification token)
    """
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    grid = tf.meshgrid(grid_w, grid_h)  # here w goes first
    grid = tf.stack(grid, axis=0)

    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = tf.concat([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = tf.range(embed_dim // 2, dtype="float32")
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = tf.reshape(pos, [-1])  # (M,)
    out = tf.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # half of the positions get sinusoidal pattern and the rest gets
    # cosine pattern and then they are concatenated
    emb_sin = tf.sin(out)  # (M, D/2)
    emb_cos = tf.cos(out)  # (M, D/2)

    emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TFViTMAEEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.patch_embeddings = TFViTMAEPatchEmbeddings(config, name="patch_embeddings")
        self.num_patches = self.patch_embeddings.num_patches

        self.config = config

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        self.position_embeddings = self.add_weight(
            shape=(1, self.num_patches + 1, self.config.hidden_size),
            initializer="zeros",
            trainable=False,  # fixed sin-cos embedding
            name="position_embeddings",
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )[None, ...]
        self.position_embeddings.assign(pos_embed)

        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, seq_len, dim = shape_list(embeddings)
        num_patches = seq_len - 1

        _, num_positions, _ = shape_list(self.position_embeddings)
        num_positions -= 1

        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bicubic",
        )

        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def random_masking(self, sequence: tf.Tensor, noise: tf.Tensor | None = None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`tf.Tensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = shape_list(sequence)
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = tf.random.uniform(shape=(batch_size, seq_length), minval=0.0, maxval=1.0)  # noise in [0, 1)

        # sort noise for each sample
        ids_shuffle = tf.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = tf.gather(
            sequence,
            axis=1,
            batch_dims=1,
            indices=ids_keep,
        )

        # generate the binary mask: 0 is keep, 1 is remove
        # this hack is needed because TF's EagerTensors don't support
        # assignment
        mask_keep = tf.zeros((batch_size, len_keep))
        mask_remove = tf.ones((batch_size, seq_length - len_keep))
        mask = tf.concat([mask_keep, mask_remove], axis=-1)

        # unshuffle to get the binary mask
        mask = tf.gather(mask, axis=1, batch_dims=1, indices=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def call(
        self, pixel_values: tf.Tensor, noise: tf.Tensor = None, interpolate_pos_encoding: bool = False
    ) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings
        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = tf.tile(cls_token, (shape_list(embeddings)[0], 1, 1))
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        return embeddings, mask, ids_restore


class TFViTMAEPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        self.projection = keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # following torch.nn.Linear
            bias_initializer="zeros",
            name="projection",
        )

    def call(
        self, pixel_values: tf.Tensor, training: bool = False, interpolate_pos_encoding: bool = False
    ) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
            if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        x = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTSelfAttention with ViT->ViTMAE
class TFViTMAESelfAttention(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTSelfOutput with ViT->ViTMAE
class TFViTMAESelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTAttention with ViT->ViTMAE
class TFViTMAEAttention(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFViTMAESelfAttention(config, name="attention")
        self.dense_output = TFViTMAESelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->ViTMAE
class TFViTMAEIntermediate(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTOutput with ViT->ViTMAE
class TFViTMAEOutput(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = hidden_states + input_tensor

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTLayer with ViT->ViTMAE
class TFViTMAELayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFViTMAEAttention(config, name="attention")
        self.intermediate = TFViTMAEIntermediate(config, name="intermediate")
        self.vit_output = TFViTMAEOutput(config, name="output")

        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(
            # in ViTMAE, layernorm is applied before self-attention
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(inputs=hidden_states)

        intermediate_output = self.intermediate(hidden_states=layer_output)

        # second residual connection is done here
        layer_output = self.vit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "vit_output", None) is not None:
            with tf.name_scope(self.vit_output.name):
                self.vit_output.build(None)
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTEncoder with ViT->ViTMAE
class TFViTMAEEncoder(keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TFViTMAELayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFViTMAEMainLayer(keras.layers.Layer):
    config_class = ViTMAEConfig

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFViTMAEEmbeddings(config, name="embeddings")
        self.encoder = TFViTMAEEncoder(config, name="encoder")
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values=pixel_values,
            training=training,
            noise=noise,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return TFViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])


class TFViTMAEPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTMAEConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"


VIT_MAE_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`ViTMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_MAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.

        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).

        interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
            Whether to interpolate the position encodings at the encoder and decoder.
"""


@add_start_docstrings(
    "The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_MAE_START_DOCSTRING,
)
class TFViTMAEModel(TFViTMAEPreTrainedModel):
    def __init__(self, config: ViTMAEConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.vit = TFViTMAEMainLayer(config, name="vit")

    def get_input_embeddings(self):
        return self.vit.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        outputs = self.vit(
            pixel_values=pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                self.vit.build(None)


class TFViTMAEDecoder(keras.layers.Layer):
    def __init__(self, config, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.decoder_embed = keras.layers.Dense(config.decoder_hidden_size, name="decoder_embed")

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = [
            TFViTMAELayer(decoder_config, name=f"decoder_layers.{j}") for j in range(config.decoder_num_hidden_layers)
        ]

        self.decoder_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="decoder_norm")
        self.decoder_pred = keras.layers.Dense(
            config.patch_size**2 * config.num_channels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="decoder_pred",
        )  # encoder to decoder
        self.config = config
        self.num_patches = num_patches

    def build(self, input_shape=None):
        self.mask_token = self.add_weight(
            shape=(1, 1, self.config.decoder_hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="mask_token",
        )
        self.decoder_pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, self.config.decoder_hidden_size),
            initializer="zeros",
            trainable=False,
            name="decoder_pos_embed",
        )
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            add_cls_token=True,
        )[None, ...]
        self.decoder_pos_embed.assign(decoder_pos_embed)

        if self.built:
            return
        self.built = True
        if getattr(self, "decoder_embed", None) is not None:
            with tf.name_scope(self.decoder_embed.name):
                self.decoder_embed.build([None, None, self.config.hidden_size])
        if getattr(self, "decoder_norm", None) is not None:
            with tf.name_scope(self.decoder_norm.name):
                self.decoder_norm.build([None, None, self.config.decoder_hidden_size])
        if getattr(self, "decoder_pred", None) is not None:
            with tf.name_scope(self.decoder_pred.name):
                self.decoder_pred.build([None, None, self.config.decoder_hidden_size])
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def interpolate_pos_encoding(self, embeddings) -> tf.Tensor:
        """
        This method is a modified version of the interpolation function for ViT-mae model at the deocder, that
        allows to interpolate the pre-trained decoder position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # [batch_size, num_patches + 1, hidden_size]
        _, num_positions, dim = shape_list(self.decoder_pos_embed)

        # -1 removes the class dimension since we later append it without interpolation
        seq_len = shape_list(embeddings)[1] - 1
        num_positions = num_positions - 1

        # Separation of class token and patch tokens
        class_pos_embed = self.decoder_pos_embed[:, :1, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]

        # interpolate the position embeddings
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(patch_pos_embed, shape=(1, 1, -1, dim)),
            size=(1, seq_len),
            method="bicubic",
        )

        # [1, seq_len, hidden_size]
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        # Adding the class token back
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding=False,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)
        # append mask tokens to sequence
        mask_tokens = tf.tile(
            self.mask_token,
            (shape_list(x)[0], shape_list(ids_restore)[1] + 1 - shape_list(x)[1], 1),
        )
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        x_ = tf.gather(x_, axis=1, batch_dims=1, indices=ids_restore)  # unshuffle
        x = tf.concat([x[:, :1, :], x_], axis=1)  # append cls token
        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        # add pos embed
        hidden_states = x + decoder_pos_embed
        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return TFViTMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)


@add_start_docstrings(
    "The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIT_MAE_START_DOCSTRING,
)
class TFViTMAEForPreTraining(TFViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = TFViTMAEMainLayer(config, name="vit")
        self.decoder = TFViTMAEDecoder(
            config,
            num_patches=self.vit.embeddings.num_patches,
            name="decoder",
        )

    def get_input_embeddings(self):
        return self.vit.get_input_embeddings()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def patchify(self, pixel_values, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)` or `(batch_size, num_channels, height, width)`):
                Pixel values.
            interpolate_pos_encoding (`bool`, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # make sure channels are last
        if shape_list(pixel_values)[1] == num_channels:
            pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # sanity checks
        if not interpolate_pos_encoding:
            tf.debugging.assert_equal(
                shape_list(pixel_values)[1],
                shape_list(pixel_values)[2],
                message="Make sure the pixel values have a squared size",
            )
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1] % patch_size,
            0,
            message="Make sure the pixel values have a size that is divisible by the patch size",
        )
        tf.debugging.assert_equal(
            shape_list(pixel_values)[3],
            num_channels,
            message=(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            ),
        )

        # patchify
        batch_size = shape_list(pixel_values)[0]
        num_patches_h = shape_list(pixel_values)[1] // patch_size
        num_patches_w = shape_list(pixel_values)[2] // patch_size
        patchified_pixel_values = tf.reshape(
            pixel_values,
            (batch_size, num_patches_h, patch_size, num_patches_w, patch_size, num_channels),
        )
        patchified_pixel_values = tf.einsum("nhpwqc->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels),
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`Tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `tf.Tensor` of shape `(batch_size, height, width, num_channels)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        tf.debugging.assert_equal(
            num_patches_h * num_patches_w,
            shape_list(patchified_pixel_values)[1],
            message=f"The number of patches in the patchified pixel values is {shape_list(patchified_pixel_values)[1]} does not match the patches of original image {num_patches_w}*{num_patches_h}",
        )

        # unpatchify
        batch_size = shape_list(patchified_pixel_values)[0]
        patchified_pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_h, num_patches_w, patch_size, patch_size, num_channels),
        )
        patchified_pixel_values = tf.einsum("nhwpqc->nhpwqc", patchified_pixel_values)
        pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_h * patch_size, num_patches_w * patch_size, num_channels),
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)`):
                Pixel values.
            pred (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `tf.Tensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.config.norm_pix_loss:
            mean = tf.reduce_mean(target, axis=-1, keepdims=True)
            var = tf.math.reduce_variance(target, axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = tf.reduce_mean(loss, axis=-1)  # [batch_size, num_patches], mean loss per patch

        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)  # mean loss on removed patches
        loss = tf.reshape(loss, (1,))
        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> Union[TFViTMAEForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFViTMAEForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = TFViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values=pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        # [batch_size, num_patches, patch_size**2*3]
        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        logits = decoder_outputs.logits

        loss = self.forward_loss(pixel_values, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                self.vit.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


__all__ = ["TFViTMAEForPreTraining", "TFViTMAEModel", "TFViTMAEPreTrainedModel"]
