# coding=utf-8
# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
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
"""TF 2.0 GroupViT model."""

from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    logging,
    replace_return_docstrings,
)
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig


logger = logging.get_logger(__name__)

# soft dependency
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp

        # On the first call, check whether a compatible version of TensorFlow is installed
        # TensorFlow Probability depends on a recent stable release of TensorFlow
        _ = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        logger.error(
            "GroupViT models are not usable since `tensorflow_probability` can't be loaded. "
            "It seems you have `tensorflow_probability` installed with the wrong tensorflow version."
            "Please try to reinstall it following the instructions here: https://github.com/tensorflow/probability."
        )
else:
    try:
        import tensorflow_probability as tfp

        # On the first call, check whether a compatible version of TensorFlow is installed
        # TensorFlow Probability depends on a recent stable release of TensorFlow
        _ = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        pass

_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# Copied from transformers.models.clip.modeling_tf_clip.clip_loss with clip->groupvit
def groupvit_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


def hard_softmax(logits: tf.Tensor, dim: int) -> tf.Tensor:
    y_soft = stable_softmax(logits, dim)
    # Straight through.
    index = tf.argmax(y_soft, dim)
    y_hard = tf.one_hot(
        index,
        depth=shape_list(logits)[dim],
        # TensorFlow expects axis to be -1 or between [0, 3).  But received: -2
        # This is why the following code snippet is used.
        axis=range(len(shape_list(logits)))[dim],
        dtype=y_soft.dtype,
    )
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft

    return ret


def gumbel_softmax(logits: tf.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> tf.Tensor:
    gumbel_dist = tfp.distributions.Gumbel(0.0, 1.0)
    gumbels = gumbel_dist.sample(tf.shape(logits), dtype=logits.dtype)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = stable_softmax(gumbels, dim)

    if hard:
        # Straight through.
        index = tf.argmax(y_soft, dim)
        y_hard = tf.one_hot(
            index,
            depth=shape_list(logits)[dim],
            # TensorFlow expects axis to be -1 or between [0, 3).  But received: -2
            # This is why the following code snippet is used.
            axis=range(len(shape_list(logits)))[dim],
            dtype=y_soft.dtype,
        )
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def resize_attention_map(attentions: tf.Tensor, height: int, width: int, align_corners: bool = False) -> tf.Tensor:
    """
    Args:
        attentions (`tf.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]
        height (`int`): height of the output attention map
        width (`int`): width of the output attention map
        align_corners (`bool`, *optional*): the `align_corner` argument for `nn.functional.interpolate`.

    Returns:
        `tf.Tensor`: resized attention map of shape [batch_size, groups, height, width]
    """

    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = shape_list(attentions)[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = shape_list(attentions)[2] // feat_height

    batch_size = shape_list(attentions)[0]
    groups = shape_list(attentions)[1]  # number of group token
    # [batch_size, groups, height x width, groups] -> [batch_size, groups, height, width]
    attentions = tf.reshape(attentions, (batch_size, groups, feat_height, feat_width))
    attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
    if align_corners:
        attentions = tf.compat.v1.image.resize(
            attentions,
            size=(height, width),
            method="bilinear",
            align_corners=align_corners,
        )
    else:
        attentions = tf.image.resize(attentions, size=(height, width), method="bilinear")
    attentions = tf.transpose(attentions, perm=(0, 3, 1, 2))
    return attentions


def get_grouping_from_attentions(attentions: Tuple[tf.Tensor], hw_shape: Tuple[int]) -> tf.Tensor:
    """
    Args:
        attentions (`tuple(tf.Tensor)`: tuple of attention maps returned by `TFGroupViTVisionTransformer`
        hw_shape (`tuple(int)`): height and width of the output attention map
    Returns:
        `tf.Tensor`: the attention map of shape [batch_size, groups, height, width]
    """

    attn_maps = []
    prev_attn_masks = None
    for attn_masks in attentions:
        # [batch_size, num_groups, height x width] -> [batch_size, height x width, num_groups]
        attn_masks = tf.transpose(attn_masks, perm=(0, 2, 1))
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            prev_attn_masks = tf.matmul(prev_attn_masks, attn_masks)
        # [batch_size, height x width, num_groups] -> [batch_size, num_groups, height x width] -> [batch_size, num_groups, height, width]
        cur_attn_map = resize_attention_map(tf.transpose(prev_attn_masks, perm=(0, 2, 1)), *hw_shape)
        attn_maps.append(cur_attn_map)

    # [batch_size, num_groups, height, width]
    final_grouping = attn_maps[-1]

    return tf.stop_gradient(final_grouping)


@dataclass
class TFGroupViTModelOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        segmentation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTTextModel`].
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTVisionModel`].
        text_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTTextModel`].
        vision_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTVisionModel`].
    """

    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    segmentation_logits: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class TFGroupViTCrossAttentionLayer(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.attn = TFGroupViTAttention(config, name="attn")
        self.norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm2")
        self.mlp = TFGroupViTMLP(config, name="mlp")
        self.norm_post = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_post")
        self.config = config

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = query
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        x = x + self.mlp(self.norm2(x))
        x = self.norm_post(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, "norm2", None) is not None:
            with tf.name_scope(self.norm2.name):
                self.norm2.build([None, None, self.config.hidden_size])
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, "norm_post", None) is not None:
            with tf.name_scope(self.norm_post.name):
                self.norm_post.build([None, None, self.config.hidden_size])


class TFGroupViTAssignAttention(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.scale = config.hidden_size**-0.5

        self.q_proj = keras.layers.Dense(config.hidden_size, name="q_proj")
        self.k_proj = keras.layers.Dense(config.hidden_size, name="k_proj")
        self.v_proj = keras.layers.Dense(config.hidden_size, name="v_proj")
        self.proj = keras.layers.Dense(config.hidden_size, name="proj")
        self.assign_eps = config.assign_eps
        self.config = config

    def get_attn(self, attn: tf.Tensor, gumbel: bool = True, hard: bool = True, training: bool = False) -> tf.Tensor:
        if gumbel and training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        else:
            if hard:
                attn = hard_softmax(attn, dim=-2)
            else:
                attn = stable_softmax(attn, axis=-2)

        return attn

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool = False):
        value = key
        # [batch_size, query_length, channels]
        query = self.q_proj(query)

        # [batch_size, key_length, channels]
        key = self.k_proj(key)

        # [batch_size, key_length, channels]
        value = self.v_proj(value)

        # [batch_size, query_length, key_length]
        raw_attn = tf.matmul(query, key, transpose_b=True) * self.scale

        attn = self.get_attn(raw_attn, training=training)
        soft_attn = self.get_attn(raw_attn, training=training, gumbel=False, hard=False)

        attn = attn / (tf.math.reduce_sum(attn, axis=-1, keepdims=True) + self.assign_eps)

        out = tf.matmul(attn, value)

        out = self.proj(out)

        return out, soft_attn

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.config.hidden_size])
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.config.hidden_size])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.config.hidden_size])
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.config.hidden_size])


class TFGroupViTTokenAssign(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, num_group_token: int, num_output_group: int, **kwargs):
        super().__init__(**kwargs)
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_tokens")
        assign_mlp_ratio = (
            config.assign_mlp_ratio
            if isinstance(config.assign_mlp_ratio, collections.abc.Iterable)
            else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        )
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        self.mlp_inter = TFGroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group, name="mlp_inter")
        self.norm_post_tokens = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_post_tokens")
        # norm on x
        self.norm_x = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_x")
        self.pre_assign_attn = TFGroupViTCrossAttentionLayer(config, name="pre_assign_attn")

        self.assign = TFGroupViTAssignAttention(config, name="assign")
        self.norm_new_x = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_new_x")
        self.mlp_channels = TFGroupViTMLP(
            config, config.hidden_size, channels_dim, config.hidden_size, name="mlp_channels"
        )
        self.config = config

    def project_group_token(self, group_tokens: tf.Tensor) -> tf.Tensor:
        """
        Args:
            group_tokens (tf.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (tf.Tensor): [batch_size, num_output_groups, channels]
        """
        # [B, num_output_groups, C] <- [B, num_group_tokens, C]
        projected_group_tokens = self.mlp_inter(group_tokens)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def call(self, image_tokens: tf.Tensor, group_tokens: tf.Tensor, training: bool = False):
        """
        Args:
            image_tokens (`tf.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`tf.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """

        group_tokens = self.norm_tokens(group_tokens)
        image_tokens = self.norm_x(image_tokens)
        # [batch_size, num_output_groups, channels]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        new_image_tokens += projected_group_tokens

        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))

        return new_image_tokens, attention

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "norm_tokens", None) is not None:
            with tf.name_scope(self.norm_tokens.name):
                self.norm_tokens.build([None, None, self.config.hidden_size])
        if getattr(self, "mlp_inter", None) is not None:
            with tf.name_scope(self.mlp_inter.name):
                self.mlp_inter.build(None)
        if getattr(self, "norm_post_tokens", None) is not None:
            with tf.name_scope(self.norm_post_tokens.name):
                self.norm_post_tokens.build([None, None, self.config.hidden_size])
        if getattr(self, "norm_x", None) is not None:
            with tf.name_scope(self.norm_x.name):
                self.norm_x.build([None, None, self.config.hidden_size])
        if getattr(self, "pre_assign_attn", None) is not None:
            with tf.name_scope(self.pre_assign_attn.name):
                self.pre_assign_attn.build(None)
        if getattr(self, "assign", None) is not None:
            with tf.name_scope(self.assign.name):
                self.assign.build(None)
        if getattr(self, "norm_new_x", None) is not None:
            with tf.name_scope(self.norm_new_x.name):
                self.norm_new_x.build([None, None, self.config.hidden_size])
        if getattr(self, "mlp_channels", None) is not None:
            with tf.name_scope(self.mlp_channels.name):
                self.mlp_channels.build(None)


# Adapted from transformers.models.vit.modeling_tf_vit.TFViTPatchEmbeddings with ViT->GroupViT
class TFGroupViTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)
        image_size, patch_size = config.image_size, config.patch_size
        num_channels = config.num_channels
        # hidden_size is a member as it will be required in the call method
        self.hidden_size = config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        self.projection = keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if (
            not interpolate_pos_encoding
            and tf.executing_eagerly()
            and (height != self.image_size[0] or width != self.image_size[1])
        ):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # When running on CPU, `keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        # In the TFGroupViTVisionEmbeddings the embeddings from this layer will be layer normalized
        # LayerNormalization layer needs to have static last dimension (otherwise the test_keras_save_load fails with symbolic tensors)
        # This is why we have used the hidden_size in the reshape method
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, self.hidden_size))

        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])


# Adapted from transformers.vit.modeling_tf_vit.TFViTEmbeddings
class TFGroupViTVisionEmbeddings(keras.layers.Layer):
    """
    Construct the position and patch embeddings.

    """

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.patch_embeddings = TFGroupViTPatchEmbeddings(config, name="patch_embeddings")
        self.dropout = keras.layers.Dropout(rate=config.dropout, name="dropout")
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.config = config

    def build(self, input_shape=None):
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches, self.config.hidden_size),
            initializer="zeros",
            trainable=True,
            name="position_embeddings",
        )

        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, num_patches, dim = shape_list(embeddings)
        num_positions = shape_list(self.position_embeddings)[1]

        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
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
        return patch_pos_embed

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        _, _, height, width = shape_list(pixel_values)
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        embeddings = self.layernorm(embeddings)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPTextEmbeddings with CLIP->GroupViT
class TFGroupViTTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size

        self.config = config

    def build(self, input_shape: tf.TensorShape = None):
        with tf.name_scope("token_embedding"):
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings


class TFGroupViTStage(keras.layers.Layer):
    """This corresponds to the `GroupingLayer` class in the GroupViT implementation."""

    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.depth = depth
        self.num_group_token = num_group_token
        self.layers = [TFGroupViTEncoderLayer(config, name=f"layers_._{i}") for i in range(depth)]

        if num_group_token > 0:
            self.downsample = TFGroupViTTokenAssign(
                config=config,
                num_group_token=num_group_token,
                num_output_group=num_output_group,
                name="downsample",
            )
        else:
            self.downsample = None

        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = [
                keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="group_projector.0"),
                TFGroupViTMixerMLP(
                    config, num_prev_group_token, config.hidden_size // 2, num_group_token, name="group_projector.1"
                ),
            ]
        else:
            self.group_projector = None

    def build(self, input_shape=None):
        if self.num_group_token > 0:
            self.group_token = self.add_weight(
                shape=(1, self.num_group_token, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                name="group_token",
            )
        else:
            self.group_token = None

        if self.built:
            return
        self.built = True
        if getattr(self, "downsample", None) is not None:
            with tf.name_scope(self.downsample.name):
                self.downsample.build(None)
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, "group_projector", None) is not None:
            with tf.name_scope(self.group_projector[0].name):
                self.group_projector[0].build([None, None, self.config.hidden_size])
            with tf.name_scope(self.group_projector[1].name):
                self.group_projector[1].build(None)

    @property
    def with_group_token(self):
        return self.group_token is not None

    def split_x(self, x: tf.Tensor) -> tf.Tensor:
        if self.with_group_token:
            return x[:, : -self.num_group_token], x[:, -self.num_group_token :]
        else:
            return x, None

    def concat_x(self, x: tf.Tensor, group_token: tf.Tensor | None = None) -> tf.Tensor:
        if group_token is None:
            return x
        return tf.concat([x, group_token], axis=1)

    def call(
        self,
        hidden_states: tf.Tensor,
        prev_group_token: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the grouping tensors of Grouping block.
        """
        if self.with_group_token:
            group_token = tf.tile(self.group_token, multiples=(shape_list(hidden_states)[0], 1, 1))
            if self.group_projector is not None:
                for layer in self.group_projector:
                    prev_group_token = layer(prev_group_token)
                group_token = group_token + prev_group_token
        else:
            group_token = None

        x = hidden_states

        cat_x = self.concat_x(x, group_token)
        for layer in self.layers:
            layer_out = layer(
                cat_x,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
            )
            cat_x = layer_out[0]

        x, group_token = self.split_x(cat_x)

        attention = None
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)

        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)

        return outputs


class TFGroupViTMLP(keras.layers.Layer):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.activation_fn = get_tf_activation(config.hidden_act)
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        output_size = output_size if output_size is not None else hidden_size
        self.fc1 = keras.layers.Dense(intermediate_size, name="fc1")
        self.fc2 = keras.layers.Dense(output_size, name="fc2")
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.hidden_size])
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.intermediate_size])


class TFGroupViTMixerMLP(TFGroupViTMLP):
    def call(self, x, training: bool = False):
        x = super().call(hidden_states=tf.transpose(x, perm=(0, 2, 1)))
        return tf.transpose(x, perm=(0, 2, 1))


# Adapted from transformers.models.clip.modeling_tf_clip.TFCLIPAttention
class TFGroupViTAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_attention_heads})."
            )

        factor = config.initializer_factor
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        out_proj_std = (self.embed_dim**-0.5) * factor

        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.q_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        self.dropout = keras.layers.Dropout(rate=config.attention_dropout)

        self.out_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor = None,
        causal_attention_mask: tf.Tensor = None,
        output_attentions: bool = None,
        encoder_hidden_states: tf.Tensor = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size = shape_list(hidden_states)[0]
        is_cross_attention = encoder_hidden_states is not None

        mixed_query_layer = self.q_proj(inputs=hidden_states)
        if is_cross_attention:
            mixed_key_layer = self.k_proj(inputs=encoder_hidden_states)
            mixed_value_layer = self.v_proj(inputs=encoder_hidden_states)
        else:
            mixed_key_layer = self.k_proj(inputs=hidden_states)
            mixed_value_layer = self.v_proj(inputs=hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            # Apply the causal attention mask (precomputed for all layers in TFCLIPModel call() function)
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in TFCLIPModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=_attention_probs)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, embed_dim)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        attention_output = self.out_proj(attention_output)
        # In TFBert, attention weights are returned after dropout.
        # However, in CLIP, they are returned before dropout.
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])


# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPEncoderLayer with CLIP->GroupViT
class TFGroupViTEncoderLayer(keras.layers.Layer):
    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.self_attn = TFGroupViTAttention(config, name="self_attn")
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFGroupViTMLP(config, name="mlp")
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            causal_attention_mask (`tf.Tensor`): causal attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`):
                Whether or not to return the attentions tensors of all attention layers. See `outputs` under returned
                tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(inputs=hidden_states)
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(inputs=hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + attention_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])


# Adapted from transformers.models.clip.modeling_tf_clip.TFGroupViTTextEncoder
class TFGroupViTTextEncoder(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.layers = [TFGroupViTEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFGroupViTVisionEncoder(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        self.stages = [
            TFGroupViTStage(
                config=config,
                depth=config.depths[i],
                num_group_token=config.num_group_tokens[i],
                num_output_group=config.num_output_groups[i],
                num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0,
                name=f"stages_._{i}",
            )
            for i in range(len(config.depths))
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_groupings = () if output_attentions else None

        group_tokens = None

        for stage in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = stage(hidden_states, group_tokens, output_attentions)

            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]

            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "stages", None) is not None:
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)


# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPTextTransformer with CLIPText->GroupViTText, CLIPEncoder->GroupViTTextEncoder
class TFGroupViTTextTransformer(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFGroupViTTextEmbeddings(config, name="embeddings")
        self.encoder = TFGroupViTTextEncoder(config, name="encoder")
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id
        self.embed_dim = config.hidden_size

    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: tf.Tensor,
        position_ids: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        input_shape = shape_list(input_ids)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        batch_size, seq_length = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # check attention mask and invert
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1
                ),
            )
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(
                        tf.range(input_shape[0], dtype=tf.int64),
                        tf.math.argmax(tf.cast(input_ids == self.eos_token_id, dtype=tf.int8), axis=-1),
                    ),
                    axis=1,
                ),
            )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        # It is possible with an unspecified sequence length for seq_length to be
        # a runtime value, which is unsupported by tf.constant. Per the TensorFlow
        # docs, tf.fill can handle runtime dynamic shapes:
        # https://www.tensorflow.org/api_docs/python/tf/fill
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)

        # set an additive 2D attention mask with all places being masked
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)

        # set diagonal & lower triangular parts to 0 (i.e. the places not to be masked)
        # TIP: think the 2D matrix as the space of (query_seq, key_seq)
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        # to_mask = tf.linalg.band_part(to_mask, -1, 0)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

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
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])


# Adapted from transformers.models.clip.modeling_tf_clip.TFCLIPVisionTransformer
class TFGroupViTVisionTransformer(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFGroupViTVisionEmbeddings(config, name="embeddings")
        self.encoder = TFGroupViTVisionEncoder(config, name="encoder")
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.embed_dim = config.hidden_size

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # normalize the last hidden state
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = tf.math.reduce_mean(last_hidden_state, axis=1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
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
                self.layernorm.build([None, None, self.embed_dim])


@keras_serializable
# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPTextMainLayer with CLIP->GroupViT
class TFGroupViTTextMainLayer(keras.layers.Layer):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFGroupViTTextTransformer(config, name="text_model")

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return text_model_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)


@keras_serializable
# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPVisionMainLayer with CLIP->GroupViT
class TFGroupViTVisionMainLayer(keras.layers.Layer):
    config_class = GroupViTVisionConfig

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vision_model = TFGroupViTVisionTransformer(config, name="vision_model")

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings

    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        vision_model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return vision_model_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)


@keras_serializable
# Adapted from transformers.models.clip.modeling_tf_clip.TFCLIPMainLayer
class TFGroupViTMainLayer(keras.layers.Layer):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(config.text_config, GroupViTTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type GroupViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type GroupViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        self.config = config

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = TFGroupViTTextTransformer(text_config, name="text_model")
        self.vision_model = TFGroupViTVisionTransformer(vision_config, name="vision_model")

        self.visual_projection = [
            keras.layers.Dense(self.projection_intermediate_dim, name="visual_projection.0"),
            keras.layers.BatchNormalization(name="visual_projection.1", momentum=0.9, epsilon=1e-5),
            keras.layers.ReLU(name="visual_projection.2"),
            keras.layers.Dense(self.projection_dim, name="visual_projection.3"),
        ]
        self.text_projection = [
            keras.layers.Dense(self.projection_intermediate_dim, name="text_projection.0"),
            keras.layers.BatchNormalization(name="text_projection.1", momentum=0.9, epsilon=1e-5),
            keras.layers.ReLU(name="text_projection.2"),
            keras.layers.Dense(self.projection_dim, name="text_projection.3"),
        ]

    def build(self, input_shape=None):
        self.logit_scale = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        if self.built:
            return
        self.built = True
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection[0].name):
                self.visual_projection[0].build([None, None, None, self.vision_embed_dim])
            with tf.name_scope(self.visual_projection[1].name):
                self.visual_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.visual_projection[3].name):
                self.visual_projection[3].build([None, None, None, self.projection_intermediate_dim])
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection[0].name):
                self.text_projection[0].build([None, None, None, self.text_embed_dim])
            with tf.name_scope(self.text_projection[1].name):
                self.text_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.text_projection[3].name):
                self.text_projection[3].build([None, None, None, self.projection_intermediate_dim])

    @unpack_inputs
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = text_outputs[1]
        for layer in self.text_projection:
            pooled_output = layer(pooled_output)

        text_features = pooled_output
        return text_features

    @unpack_inputs
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = vision_outputs[1]
        for layer in self.visual_projection:
            pooled_output = layer(pooled_output)

        image_features = pooled_output
        return image_features

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        if output_segmentation:
            output_attentions = True
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        image_embeds = vision_outputs[1]
        for layer in self.visual_projection:
            image_embeds = layer(image_embeds)

        text_embeds = text_outputs[1]
        for layer in self.text_projection:
            text_embeds = layer(text_embeds)

        # normalized features
        image_embeds = image_embeds / tf.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        seg_logits = None
        if output_segmentation:
            # grouped features
            # [batch_size_image, num_group, hidden_size]
            image_group_embeds = vision_outputs[0]
            # [batch_size_image*num_group, hidden_size]
            image_group_embeds = tf.reshape(image_group_embeds, shape=(-1, shape_list(image_group_embeds)[-1]))
            for layer in self.visual_projection:
                image_group_embeds = layer(image_group_embeds)
            if output_hidden_states:
                attentions = vision_outputs[3]
            else:
                attentions = vision_outputs[2]
            # [batch_size_image, num_group, height, width]
            grouping = get_grouping_from_attentions(attentions, pixel_values.shape[2:])

            # normalized features
            image_group_embeds = image_group_embeds / tf.norm(
                tensor=image_group_embeds, ord="euclidean", axis=-1, keepdims=True
            )
            # [batch_size_image x num_group, batch_size_text]
            logits_per_image_group = tf.matmul(image_group_embeds, text_embeds, transpose_b=True) * logit_scale
            # [batch_size_image, batch_size_text, num_group]
            logits_per_image_group = tf.reshape(
                logits_per_image_group, shape=(image_embeds.shape[0], -1, text_embeds.shape[0])
            )
            logits_per_image_group = tf.transpose(logits_per_image_group, perm=(0, 2, 1))

            # [batch_size_image, batch_size_text, height x width]
            flatten_grouping = tf.reshape(grouping, shape=(shape_list(grouping)[0], shape_list(grouping)[1], -1))

            # [batch_size_image, batch_size_text, height, width]
            seg_logits = tf.matmul(logits_per_image_group, flatten_grouping) * logit_scale
            seg_logits = tf.reshape(
                seg_logits, shape=(seg_logits.shape[0], seg_logits.shape[1], grouping.shape[2], grouping.shape[3])
            )

        loss = None
        if return_loss:
            loss = groupvit_loss(logits_per_text)[None, ...]

        if not return_dict:
            if seg_logits is not None:
                output = (
                    logits_per_image,
                    logits_per_text,
                    seg_logits,
                    text_embeds,
                    image_embeds,
                    text_outputs,
                    vision_outputs,
                )
            else:
                output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return TFGroupViTModelOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            segmentation_logits=seg_logits,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class TFGroupViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GroupViTConfig
    base_model_prefix = "groupvit"


GROUPVIT_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
    first positional argument :

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
      `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
      `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Args:
        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUPVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

GROUPVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

GROUPVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


class TFGroupViTTextModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTTextConfig
    main_input_name = "input_ids"

    def __init__(self, config: GroupViTTextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.groupvit = TFGroupViTTextMainLayer(config, name="groupvit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, TFGroupViTTextModel

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        outputs = self.groupvit(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)


class TFGroupViTVisionModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: GroupViTVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.groupvit = TFGroupViTVisionMainLayer(config, name="groupvit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTVisionModel

        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        outputs = self.groupvit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)


@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class TFGroupViTModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.groupvit = TFGroupViTMainLayer(config, name="groupvit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFGroupViTTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, TFGroupViTModel

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        text_features = self.groupvit.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return text_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFGroupViTVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTModel

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""

        image_features = self.groupvit.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFGroupViTModelOutput, config_class=GroupViTConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTModel
        >>> import tensorflow as tf

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.math.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""

        outputs = self.groupvit(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_segmentation=output_segmentation,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output: TFGroupViTModelOutput) -> TFGroupViTModelOutput:
        # TODO: As is this currently fails with saved_model=True, because
        # TensorFlow cannot trace through nested dataclasses. Reference:
        # https://github.com/huggingface/transformers/pull/16886
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)


__all__ = ["TFGroupViTModel", "TFGroupViTPreTrainedModel", "TFGroupViTTextModel", "TFGroupViTVisionModel"]
