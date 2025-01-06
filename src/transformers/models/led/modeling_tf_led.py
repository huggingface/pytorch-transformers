# coding=utf-8
# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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
"""TF 2.0 LED model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions

# Public API
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
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_led import LEDConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "allenai/led-base-16384"
_CONFIG_FOR_DOC = "LEDConfig"


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart.shift_tokens_right
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # "Verify that `labels` has only positive values and -100"
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # Make sure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


# Copied from transformers.models.bart.modeling_tf_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


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


class TFLEDLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        seq_len = input_shape[1]
        position_ids = tf.range(seq_len, delta=1, name="range")
        position_ids += past_key_values_length

        return super().call(tf.cast(position_ids, dtype=tf.int32))


# Copied from transformers.models.longformer.modeling_tf_longformer.TFLongformerSelfAttention with TFLongformer->TFLEDEncoder
class TFLEDEncoderSelfAttention(keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # separate projection layers for tokens with global attention
        self.query_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_global",
        )
        self.key_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_global",
        )
        self.value_global = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_global",
        )
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]

        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def build(self, input_shape=None):
        if not self.built:
            with tf.name_scope("query_global"):
                self.query_global.build((self.config.hidden_size,))
            with tf.name_scope("key_global"):
                self.key_global.build((self.config.hidden_size,))
            with tf.name_scope("value_global"):
                self.value_global.build((self.config.hidden_size,))

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
        if getattr(self, "query_global", None) is not None:
            with tf.name_scope(self.query_global.name):
                self.query_global.build([None, None, self.config.hidden_size])
        if getattr(self, "key_global", None) is not None:
            with tf.name_scope(self.key_global.name):
                self.key_global.build([None, None, self.config.hidden_size])
        if getattr(self, "value_global", None) is not None:
            with tf.name_scope(self.value_global.name):
                self.value_global.build([None, None, self.config.hidden_size])

    def call(
        self,
        inputs,
        training=False,
    ):
        """
        LongformerSelfAttention expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        # retrieve input args
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        batch_size, seq_len, embed_dim = shape_list(hidden_states)

        tf.debugging.assert_equal(
            embed_dim,
            self.embed_dim,
            message=f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}",
        )

        # normalize query
        query_vectors /= tf.math.sqrt(tf.cast(self.head_dim, dtype=query_vectors.dtype))
        query_vectors = tf.reshape(query_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))
        key_vectors = tf.reshape(key_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))

        # attn_probs = (batch_size, seq_len, num_heads, window*2+1)
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = attention_mask != 0
        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = tf.cast(remove_from_windowed_attention_mask, dtype=query_vectors.dtype) * LARGE_NEGATIVE

        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            tf.ones(shape_list(attention_mask)),
            float_mask,
            self.one_sided_attn_window_size,
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        tf.debugging.assert_equal(
            shape_list(attn_scores),
            [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1],
            message=(
                f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
                f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {shape_list(attn_scores)}"
            ),
        )

        # compute global attn indices required through out forward fn
        (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        ) = self._get_global_attn_indices(is_index_global_attn)

        # this function is only relevant for global attention
        if is_global_attn:
            attn_scores = self._concat_with_global_key_attn_probs(
                attn_scores=attn_scores,
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )

        attn_probs = stable_softmax(attn_scores, axis=-1)

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        # Make sure to create a mask with the proper shape:
        # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
        # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
        if is_global_attn:
            masked_index = tf.tile(
                is_index_masked[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
            )
        else:
            masked_index = tf.tile(
                is_index_masked[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
            )
        attn_probs = tf.where(
            masked_index,
            tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype),
            attn_probs,
        )

        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.num_heads],
                message=(
                    f"Head mask for a single layer should be of size {(self.num_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )

            attn_probs = tf.reshape(layer_head_mask, (1, 1, -1, 1)) * attn_probs

        # apply dropout
        attn_probs = self.dropout(attn_probs, training=training)
        value_vectors = tf.reshape(value_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))

        # if global attention, compute sum of global and local attn

        if is_global_attn:
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        tf.debugging.assert_equal(
            shape_list(attn_output), [batch_size, seq_len, self.num_heads, self.head_dim], message="Unexpected size"
        )

        attn_output = tf.reshape(attn_output, (batch_size, seq_len, embed_dim))

        # compute value for global attention and overwrite to attention output
        if is_global_attn:
            attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                attn_output=attn_output,
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
                training=training,
            )
        else:
            # Leave attn_output unchanged
            global_attn_probs = tf.zeros((batch_size, self.num_heads, max_num_global_attn_indices, seq_len))

        # make sure that local attention probabilities are set to 0 for indices of global attn
        # Make sure to create a mask with the proper shape:
        # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
        # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
        if is_global_attn:
            masked_global_attn_index = tf.tile(
                is_index_global_attn[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
            )
        else:
            masked_global_attn_index = tf.tile(
                is_index_global_attn[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
            )
        attn_probs = tf.where(
            masked_global_attn_index,
            tf.zeros(shape_list(masked_global_attn_index), dtype=attn_probs.dtype),
            attn_probs,
        )

        outputs = (attn_output, attn_probs, global_attn_probs)

        return outputs

    def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = shape_list(query)

        tf.debugging.assert_equal(
            seq_len % (window_overlap * 2),
            0,
            message=f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}",
        )
        tf.debugging.assert_equal(
            shape_list(query),
            shape_list(key),
            message=(
                f"Shape of query and key should be equal, but got query: {shape_list(query)} and key:"
                f" {shape_list(key)}"
            ),
        )

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = tf.reshape(
            tf.transpose(query, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )
        key = tf.reshape(tf.transpose(key, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        chunked_query = tf.cast(chunked_query, dtype=chunked_key.dtype)
        chunked_attention_scores = tf.einsum("bcxd,bcyd->bcxy", chunked_query, chunked_key)  # multiply

        # convert diagonals into columns
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 1], [0, 0]])
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(chunked_attention_scores, paddings)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        # TODO: This code is most likely not very efficient and should be improved
        diagonal_attn_scores_up_triang = tf.concat(
            [
                diagonal_chunked_attention_scores[:, :, :window_overlap, : window_overlap + 1],
                diagonal_chunked_attention_scores[:, -1:, window_overlap:, : window_overlap + 1],
            ],
            axis=1,
        )

        # - copying the lower triangle
        diagonal_attn_scores_low_triang = tf.concat(
            [
                tf.zeros(
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                    dtype=diagonal_chunked_attention_scores.dtype,
                ),
                diagonal_chunked_attention_scores[:, :, -(window_overlap + 1) : -1, window_overlap + 1 :],
            ],
            axis=1,
        )
        diagonal_attn_scores_first_chunk = tf.concat(
            [
                tf.roll(
                    diagonal_chunked_attention_scores,
                    shift=[1, window_overlap],
                    axis=[2, 3],
                )[:, :, :window_overlap, :window_overlap],
                tf.zeros(
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                    dtype=diagonal_chunked_attention_scores.dtype,
                ),
            ],
            axis=1,
        )
        first_chunk_mask = (
            tf.tile(
                tf.range(chunks_count + 1, dtype=tf.int64)[None, :, None, None],
                (batch_size * num_heads, 1, window_overlap, window_overlap),
            )
            < 1
        )
        diagonal_attn_scores_low_triang = tf.where(
            first_chunk_mask,
            diagonal_attn_scores_first_chunk,
            diagonal_attn_scores_low_triang,
        )

        # merging upper and lower triangle
        diagonal_attention_scores = tf.concat(
            [diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang], axis=-1
        )

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = tf.transpose(
            tf.reshape(
                diagonal_attention_scores,
                (batch_size, num_heads, seq_len, 2 * window_overlap + 1),
            ),
            (0, 2, 1, 3),
        )

        diagonal_attention_scores = self._mask_invalid_locations(diagonal_attention_scores, window_overlap)

        return diagonal_attention_scores

    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # create correct upper triangle bool mask
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )

        # pad to full matrix
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )

        # create lower mask
        mask_2d = tf.pad(mask_2d_upper, padding)

        # combine with upper mask
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

        # broadcast to full matrix
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))

        # inf tensor used for masking
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)

        # mask
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

        return input_tensor

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """

        batch_size, seq_len, num_heads, head_dim = shape_list(value)

        tf.debugging.assert_equal(
            seq_len % (window_overlap * 2), 0, message="Seq_len has to be multiple of 2 * window_overlap"
        )
        tf.debugging.assert_equal(
            shape_list(attn_probs)[:3],
            shape_list(value)[:3],
            message="value and attn_probs must have same dims (except head_dim)",
        )
        tf.debugging.assert_equal(
            shape_list(attn_probs)[3],
            2 * window_overlap + 1,
            message="attn_probs last dim has to be 2 * window_overlap + 1",
        )

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )

        # group batch_size and num_heads dimensions into one
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count
        chunked_value = tf.signal.frame(
            tf.reshape(padded_value, (batch_size * num_heads, -1)),
            frame_size,
            frame_hop_size,
        )
        chunked_value = tf.reshape(
            chunked_value,
            (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim),
        )

        tf.debugging.assert_equal(
            shape_list(chunked_value),
            [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
            message="Chunked value has the wrong shape",
        )

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )

        return context

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # padding value is not important because it will be overwritten
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # define frame size and frame stride (similar to convolution)
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # chunk with overlap
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        tf.debugging.assert_equal(
            shape_list(chunked_hidden_states),
            [batch_size, num_output_chunks, frame_size],
            message=(
                "Make sure chunking is correctly applied. `Chunked hidden states should have output  dimension"
                f" {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}."
            ),
        )

        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)

        # max number of global attn indices in batch
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)

        # indices of global attn
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)

        # helper variable
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        attn_scores,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = shape_list(key_vectors)[0]

        # select global key vectors
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

        # create only global key vectors
        key_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_key_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

        # scatter mask
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # concat to attn_probs
        # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        return attn_scores

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = shape_list(attn_probs)[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

        # select global value vectors
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)

        # create only global value vectors
        value_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_value_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # compute attn output only global
        attn_output_only_global = tf.einsum("blhs,bshd->blhd", attn_probs_only_global, value_vectors_only_global)

        # reshape attn probs
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )

        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        attn_output,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
        training,
    ):
        batch_size, seq_len = shape_list(hidden_states)[:2]

        # prepare global hidden states
        global_attn_hidden_states = tf.gather_nd(hidden_states, is_index_global_attn_nonzero)
        global_attn_hidden_states = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_attn_hidden_states,
            shape=(batch_size, max_num_global_attn_indices, self.embed_dim),
        )

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= tf.math.sqrt(
            tf.cast(self.head_dim, dtype=global_query_vectors_only_global.dtype)
        )
        global_query_vectors_only_global = self.reshape_and_transpose(global_query_vectors_only_global, batch_size)
        global_key_vectors = self.reshape_and_transpose(global_key_vectors, batch_size)
        global_value_vectors = self.reshape_and_transpose(global_value_vectors, batch_size)

        # compute attn scores
        global_attn_scores = tf.matmul(global_query_vectors_only_global, global_key_vectors, transpose_b=True)

        tf.debugging.assert_equal(
            shape_list(global_attn_scores),
            [batch_size * self.num_heads, max_num_global_attn_indices, seq_len],
            message=(
                "global_attn_scores have the wrong size. Size should be"
                f" {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is"
                f" {shape_list(global_attn_scores)}."
            ),
        )

        global_attn_scores = tf.reshape(
            global_attn_scores,
            (batch_size, self.num_heads, max_num_global_attn_indices, seq_len),
        )
        global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(global_attn_scores_trans)[-2:]
        )
        global_attn_mask = tf.ones(mask_shape) * -10000.0
        global_attn_mask = tf.cast(global_attn_mask, dtype=global_attn_scores_trans.dtype)

        # scatter mask
        global_attn_scores_trans = tf.tensor_scatter_nd_update(
            global_attn_scores_trans,
            is_local_index_no_global_attn_nonzero,
            global_attn_mask,
        )
        global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))

        # mask global attn scores
        attn_mask = tf.tile(is_index_masked[:, None, None, :], (1, shape_list(global_attn_scores)[1], 1, 1))
        global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)
        global_attn_scores = tf.reshape(
            global_attn_scores,
            (batch_size * self.num_heads, max_num_global_attn_indices, seq_len),
        )

        # compute global attn probs
        global_attn_probs_float = stable_softmax(global_attn_scores, axis=-1)

        # apply layer head masking
        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.num_heads],
                message=(
                    f"Head mask for a single layer should be of size {(self.num_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )
            global_attn_probs_float = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                global_attn_probs_float, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            )
            global_attn_probs_float = tf.reshape(
                global_attn_probs_float, (batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
            )

        # dropout
        global_attn_probs = self.global_dropout(global_attn_probs_float, training=training)

        # global attn output
        global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)

        tf.debugging.assert_equal(
            shape_list(global_attn_output),
            [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim],
            message=(
                "global_attn_output tensor has the wrong size. Size should be"
                f" {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is"
                f" {shape_list(global_attn_output)}."
            ),
        )

        global_attn_output = tf.reshape(
            global_attn_output,
            (batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim),
        )

        # get only non zero global attn output
        nonzero_global_attn_output = tf.gather_nd(
            tf.transpose(global_attn_output, (0, 2, 1, 3)),
            is_local_index_global_attn_nonzero,
        )
        nonzero_global_attn_output = tf.reshape(
            nonzero_global_attn_output,
            (shape_list(is_local_index_global_attn_nonzero)[0], -1),
        )

        # overwrite values with global attention
        attn_output = tf.tensor_scatter_nd_update(
            attn_output, is_index_global_attn_nonzero, nonzero_global_attn_output
        )

        global_attn_probs = tf.reshape(
            global_attn_probs, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        )

        return attn_output, global_attn_probs

    def reshape_and_transpose(self, vector, batch_size):
        return tf.reshape(
            tf.transpose(
                tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, -1, self.head_dim),
        )


class TFLEDEncoderAttention(keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        self.longformer_self_attn = TFLEDEncoderSelfAttention(config, layer_id=layer_id, name="longformer_self_attn")
        self.output_dense = keras.layers.Dense(config.d_model, use_bias=True, name="output")
        self.config = config

    def call(self, inputs, training=False):
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs

        self_outputs = self.longformer_self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )

        attention_output = self.output_dense(self_outputs[0], training=training)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "longformer_self_attn", None) is not None:
            with tf.name_scope(self.longformer_self_attn.name):
                self.longformer_self_attn.build(None)
        if getattr(self, "output_dense", None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.d_model])


class TFLEDDecoderAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training=False,
    ) -> Tuple[tf.Tensor, tf.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        tf.debugging.assert_equal(
            shape_list(attn_weights),
            [bsz * self.num_heads, tgt_len, src_len],
            message=(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {shape_list(attn_weights)}"
            ),
        )

        if attention_mask is not None:
            tf.debugging.assert_equal(
                shape_list(attention_mask),
                [bsz, 1, tgt_len, src_len],
                message=(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {shape_list(attention_mask)}"
                ),
            )

            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + tf.cast(
                attention_mask, dtype=attn_weights.dtype
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = stable_softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.num_heads],
                message=(
                    f"Head mask for a single layer should be of size {(self.num_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )

            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                attn_weights, (bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_probs = self.dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_probs, value_states)

        tf.debugging.assert_equal(
            shape_list(attn_output),
            [bsz * self.num_heads, tgt_len, self.head_dim],
            message=(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {shape_list(attn_output)}"
            ),
        )

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])


class TFLEDEncoderLayer(keras.layers.Layer):
    def __init__(self, config: LEDConfig, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFLEDEncoderAttention(config, layer_id, name="self_attn")
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        is_index_masked: tf.Tensor,
        is_index_global_attn: tf.Tensor,
        is_global_attn: bool,
        training=False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(config.encoder_attention_heads,)*.
        """
        residual = hidden_states
        layer_outputs = self.self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )

        hidden_states = layer_outputs[0]

        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states,) + layer_outputs[1:]

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])


class TFLEDDecoderLayer(keras.layers.Layer):
    def __init__(self, config: LEDConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFLEDDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TFLEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        self.config = config

    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        encoder_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            encoder_hidden_states (`tf.Tensor`):
                cross attention input to the layer of shape *(batch, seq_len, embed_dim)*
            encoder_attention_mask (`tf.Tensor`): encoder attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(config.encoder_attention_heads,)*.
            encoder_layer_head_mask (`tf.Tensor`): mask for encoder attention heads in a given layer of
                size *(config.encoder_attention_heads,)*.
            past_key_value (`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        residual = hidden_states

        # Self-Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
            )
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return (
            hidden_states,
            self_attn_weights,
            cross_attn_weights,
            present_key_value,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])


class TFLEDPreTrainedModel(TFPreTrainedModel):
    config_class = LEDConfig
    base_model_prefix = "led"

    @property
    def input_signature(self):
        sig = super().input_signature
        sig["global_attention_mask"] = tf.TensorSpec((None, None), tf.int32, name="global_attention_mask")
        return sig


@dataclass
# Copied from transformers.models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput with TFLongformer->TFLEDEncoder
class TFLEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
            is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None
    global_attentions: Tuple[tf.Tensor, ...] | None = None


@dataclass
class TFLEDSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
            is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    decoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    decoder_attentions: Tuple[tf.Tensor, ...] | None = None
    cross_attentions: Tuple[tf.Tensor, ...] | None = None
    encoder_last_hidden_state: tf.Tensor | None = None
    encoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    encoder_attentions: Tuple[tf.Tensor, ...] | None = None
    encoder_global_attentions: Tuple[tf.Tensor, ...] | None = None


@dataclass
class TFLEDSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x`
            is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    decoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    decoder_attentions: Tuple[tf.Tensor, ...] | None = None
    cross_attentions: Tuple[tf.Tensor, ...] | None = None
    encoder_last_hidden_state: tf.Tensor | None = None
    encoder_hidden_states: Tuple[tf.Tensor, ...] | None = None
    encoder_attentions: Tuple[tf.Tensor, ...] | None = None
    encoder_global_attentions: Tuple[tf.Tensor, ...] | None = None


LED_START_DOCSTRING = r"""
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

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`LEDConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

LED_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`LedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            LED uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
        decoder_attention_mask (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
        head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tf.Tensor`, *optional*):
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            of shape `(batch_size, sequence_length, hidden_size)` is a sequence of
        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
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
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@keras_serializable
class TFLEDEncoder(keras.layers.Layer):
    config_class = LEDConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a
    [`TFLEDEncoderLayer`].

    Args:
        config: LEDConfig
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)
        if config.encoder_layerdrop > 0:
            logger.warning("Layerdrop is currently disabled in TFLED models.")
        self.layerdrop = 0.0
        self.padding_idx = config.pad_token_id

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.attention_window = config.attention_window
        self.embed_tokens = embed_tokens
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_encoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFLEDEncoderLayer(config, i, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        self.embed_dim = config.d_model

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        """
        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = attention_mask * tf.cast((global_attention_mask + 1), dtype=attention_mask.dtype)

        padding_len, input_ids, attention_mask, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.padding_idx,
        )

        input_shape = shape_list(attention_mask)
        # is index masked or global attention
        is_index_masked = tf.math.less(tf.cast(attention_mask, tf.int8), 1)
        is_index_global_attn = tf.math.greater(tf.cast(attention_mask, tf.int8), 1)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # check attention mask and invert
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask)[:, 0, 0, :]
            attention_mask = attention_mask[:, :, None, None]

        encoder_states = () if output_hidden_states else None
        all_attentions = all_global_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(head_mask)[0],
                len(self.layers),
                message=(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {shape_list(head_mask)[0]}."
                ),
            )

        # encoder layers
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                hidden_states_to_add = self.compute_hidden_states(hidden_states, padding_len)
                encoder_states = encoder_states + (hidden_states_to_add,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):  # skip the layer
                continue

            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (tf.transpose(layer_outputs[1], (0, 2, 1, 3)),)

                # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                all_global_attentions = all_global_attentions + (tf.transpose(layer_outputs[2], (0, 1, 3, 2)),)

        # undo padding
        # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
        hidden_states = self.compute_hidden_states(hidden_states, padding_len)

        # undo padding
        if output_attentions:
            all_attentions = (
                tuple([state[:, :, :-padding_len, :] for state in all_attentions])
                if padding_len > 0
                else all_attentions
            )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFLEDEncoderBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )

    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len):
        return hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states

    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        pad_token_id,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        # padding
        attention_window = (
            self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"

        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )

        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)

        if inputs_embeds is not None:
            if padding_len > 0:
                input_ids_padding = tf.fill((batch_size, padding_len), pad_token_id)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)  # no attention on the padding tokens

        return (
            padding_len,
            input_ids,
            attention_mask,
            inputs_embeds,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFLEDDecoder(keras.layers.Layer):
    config_class = LEDConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFLEDDecoderLayer`]

    Args:
        config: LEDConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        if config.decoder_layerdrop > 0:
            logger.warning("Layerdrop is currently disabled in TFLED models.")
        self.layerdrop = 0.0
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_decoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFLEDDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        self.dropout = keras.layers.Dropout(config.dropout)

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        r"""
        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`tf.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`tf.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            encoder_head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding. If `past_key_values` are used, the user can optionally input only the last
                `decoder_input_ids` (those that don't have their past key value states given to this model) of shape
                `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
                inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = shape_list(past_key_values[0][0])[2] if past_key_values is not None else 0

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(
                tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1]
            )

        if attention_mask is not None and input_shape[-1] > 1:
            combined_attention_mask = combined_attention_mask + _expand_mask(attention_mask, tgt_len=input_shape[-1])

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, tgt_len=input_shape[-1])

        hidden_states = self.layernorm_embedding(hidden_states + positions)
        hidden_states = self.dropout(hidden_states, training=training)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        all_cross_attentions = ()
        present_key_values = ()

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(head_mask)[0],
                len(self.layers),
                message=(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {shape_list(head_mask)[0]}."
                ),
            )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)

            if training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            hidden_states, layer_self_attn, layer_cross_attn, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                encoder_layer_head_mask=encoder_head_mask[idx] if encoder_head_mask is not None else None,
                past_key_value=past_key_value,
            )

            if use_cache:
                present_key_values += (present_key_value,)

            if output_attentions:
                all_self_attns += (layer_self_attn,)
                all_cross_attentions += (layer_cross_attn,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        else:
            all_hidden_states = None

        all_self_attns = all_self_attns if output_attentions else None
        all_cross_attentions = all_cross_attentions if output_attentions else None

        present_key_values = present_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, present_key_values, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        else:
            return TFBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                cross_attentions=all_cross_attentions,
            )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFLEDMainLayer(keras.layers.Layer):
    config_class = LEDConfig

    def __init__(self, config: LEDConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="led.shared",
        )
        # Additional attribute to specify the expected name scope of the layer (for loading/storing weights)
        self.shared.load_weight_prefix = "led.shared"

        self.encoder = TFLEDEncoder(config, self.shared, name="encoder")
        self.decoder = TFLEDDecoder(config, self.shared, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]] = None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a TFLEDEncoderBaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, TFLEDEncoderBaseModelOutput):
            encoder_outputs = TFLEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # If the user passed a TFLEDEncoderBaseModelOutput for encoder_outputs, we wrap it in a tuple when return_dict=False
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return TFLEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_global_attentions=encoder_outputs.global_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # The shared/tied weights expect to be in the model base namespace
        # Adding "/" to the end (not the start!) of a tf.name_scope puts it in the root namespace rather than
        # the current one.
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            self.shared.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


@add_start_docstrings(
    "The bare LED Model outputting raw hidden-states without any specific head on top.",
    LED_START_DOCSTRING,
)
class TFLEDModel(TFLEDPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.led = TFLEDMainLayer(config, name="led")

    def get_encoder(self):
        return self.led.encoder

    def get_decoder(self):
        return self.led.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLEDSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        encoder_outputs: tf.Tensor | None = None,
        global_attention_mask: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,
    ) -> Tuple[tf.Tensor] | TFLEDSeq2SeqModelOutput:
        outputs = self.led(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        return TFLEDSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            encoder_global_attentions=enc_g_attns,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "led", None) is not None:
            with tf.name_scope(self.led.name):
                self.led.build(None)


# Copied from transformers.models.bart.modeling_tf_bart.BiasLayer
class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # Note: the name of this variable will NOT be scoped when serialized, i.e. it will not be in the format of
        # "outer_layer/inner_layer/.../name:0". Instead, it will be "name:0". For further details, see:
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        return x + self.bias


@add_start_docstrings(
    "The LED Model with a language modeling head. Can be used for summarization.",
    LED_START_DOCSTRING,
)
class TFLEDForConditionalGeneration(TFLEDPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"led.encoder.embed_tokens.weight",
        r"led.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.led = TFLEDMainLayer(config, name="led")
        self.use_cache = config.use_cache
        # final_bias_logits is registered as a buffer in pytorch, so not trainable for the sake of consistency.
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

        # TODO (Joao): investigate why LED has numerical issues in XLA generate
        self.supports_xla_generation = False

    def get_decoder(self):
        return self.led.decoder

    def get_encoder(self):
        return self.led.encoder

    def get_bias(self):
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # Replaces the existing layers containing bias for correct (de)serialization.
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLEDSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: TFLEDEncoderBaseModelOutput | None = None,
        global_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor] | TFLEDSeq2SeqLMOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFLEDForConditionalGeneration
        >>> import tensorflow as tf

        >>> mname = "allenai/led-base-16384"
        >>> tokenizer = AutoTokenizer.from_pretrained(mname)
        >>> TXT = "My friends are <mask> but they eat too many carbs."
        >>> model = TFLEDForConditionalGeneration.from_pretrained(mname)
        >>> batch = tokenizer([TXT], return_tensors="tf")
        >>> logits = model(inputs=batch.input_ids).logits
        >>> probs = tf.nn.softmax(logits[0])
        >>> # probs[5] is associated with the mask token
        ```"""

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        lm_logits = tf.matmul(outputs[0], self.led.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFLEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
            encoder_global_attentions=outputs.encoder_global_attentions,
        )

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        return TFLEDSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            encoder_global_attentions=enc_g_attns,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def hf_compute_loss(self, labels, logits):
        """CrossEntropyLoss that ignores pad tokens"""
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            melted_labels = tf.reshape(labels, (-1,))
            active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(melted_labels, active_loss)
            return loss_fn(labels, reduced_logits)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only non-padding labels affect the loss
        loss_mask = tf.cast(labels != self.config.pad_token_id, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "led", None) is not None:
            with tf.name_scope(self.led.name):
                self.led.build(None)
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)


__all__ = ["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"]
