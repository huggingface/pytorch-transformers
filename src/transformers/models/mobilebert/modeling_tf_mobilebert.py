# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""TF 2.0 MobileBERT model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
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
from .configuration_mobilebert import MobileBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/mobilebert-uncased"
_CONFIG_FOR_DOC = "MobileBertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "vumichien/mobilebert-finetuned-ner"
_TOKEN_CLASS_EXPECTED_OUTPUT = "['I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']"
_TOKEN_CLASS_EXPECTED_LOSS = 0.03

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "vumichien/mobilebert-uncased-squad-v2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 3.98
_QA_TARGET_START_INDEX = 12
_QA_TARGET_END_INDEX = 13

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "vumichien/emo-mobilebert"
_SEQ_CLASS_EXPECTED_OUTPUT = "'others'"
_SEQ_CLASS_EXPECTED_LOSS = "4.72"


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPreTrainingLoss
class TFMobileBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """

    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # make sure only labels that are not equal to -100
        # are taken into account for the loss computation
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels["next_sentence_label"]), y_pred=logits[1])
        ns_loss_mask = tf.cast(labels["next_sentence_label"] != -100, dtype=unmasked_ns_loss.dtype)
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        reduced_masked_ns_loss = tf.reduce_sum(masked_ns_loss) / tf.reduce_sum(ns_loss_mask)

        return tf.reshape(reduced_masked_lm_loss + reduced_masked_ns_loss, (1,))


class TFMobileBertIntermediate(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(config.intermediate_size, name="dense")

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])


class TFLayerNorm(keras.layers.LayerNormalization):
    def __init__(self, feat_size, *args, **kwargs):
        self.feat_size = feat_size
        super().__init__(*args, **kwargs)

    def build(self, input_shape=None):
        super().build([None, None, self.feat_size])


class TFNoNorm(keras.layers.Layer):
    def __init__(self, feat_size, epsilon=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_size = feat_size

    def build(self, input_shape):
        self.bias = self.add_weight("bias", shape=[self.feat_size], initializer="zeros")
        self.weight = self.add_weight("weight", shape=[self.feat_size], initializer="ones")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor):
        return inputs * self.weight + self.bias


NORM2FN = {"layer_norm": TFLayerNorm, "no_norm": TFNoNorm}


class TFMobileBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.embedding_transformation = keras.layers.Dense(config.hidden_size, name="embedding_transformation")

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.embedded_input_size = self.embedding_size * (3 if self.trigram_input else 1)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "embedding_transformation", None) is not None:
            with tf.name_scope(self.embedding_transformation.name):
                self.embedding_transformation.build([None, None, self.embedded_input_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)

    def call(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = tf.concat(
                [
                    tf.pad(inputs_embeds[:, 1:], ((0, 0), (0, 1), (0, 0))),
                    inputs_embeds,
                    tf.pad(inputs_embeds[:, :-1], ((0, 0), (1, 0), (0, 0))),
                ],
                axis=2,
            )

        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


class TFMobileBertSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=False
    ):
        batch_size = shape_list(attention_mask)[0]
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], dtype=attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFMobileBertModel call() function)
            attention_mask = tf.cast(attention_mask, dtype=attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.true_hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.true_hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build(
                    [
                        None,
                        None,
                        self.config.true_hidden_size
                        if self.config.use_bottleneck_attention
                        else self.config.hidden_size,
                    ]
                )


class TFMobileBertSelfOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.dense = keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        if not self.use_bottleneck:
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states, residual_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFMobileBertAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self = TFMobileBertSelfAttention(config, name="self")
        self.mobilebert_output = TFMobileBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        layer_input,
        attention_mask,
        head_mask,
        output_attentions,
        training=False,
    ):
        self_outputs = self.self(
            query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=training
        )

        attention_output = self.mobilebert_output(self_outputs[0], layer_input, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)


class TFOutputBottleneck(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.hidden_size, name="dense")
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states, residual_tensor, training=False):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs, training=training)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFMobileBertOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.dense = keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        if not self.use_bottleneck:
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = TFOutputBottleneck(config, name="bottleneck")
        self.config = config

    def call(self, hidden_states, residual_tensor_1, residual_tensor_2, training=False):
        hidden_states = self.dense(hidden_states)
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
        else:
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
            hidden_states = self.bottleneck(hidden_states, residual_tensor_2)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)


class TFBottleneckLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.intra_bottleneck_size, name="dense")
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.config = config

    def call(self, inputs):
        hidden_states = self.dense(inputs)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFBottleneck(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.bottleneck_input = TFBottleneckLayer(config, name="input")
        if self.key_query_shared_bottleneck:
            self.attention = TFBottleneckLayer(config, name="attention")

    def call(self, hidden_states):
        # This method can return three different tuples of values. These different values make use of bottlenecks,
        # which are linear layers used to project the hidden states to a lower-dimensional vector, reducing memory
        # usage. These linear layer have weights that are learned during training.
        #
        # If `config.use_bottleneck_attention`, it will return the result of the bottleneck layer four times for the
        # key, query, value, and "layer input" to be used by the attention layer.
        # This bottleneck is used to project the hidden. This last layer input will be used as a residual tensor
        # in the attention self output, after the attention scores have been computed.
        #
        # If not `config.use_bottleneck_attention` and `config.key_query_shared_bottleneck`, this will return
        # four values, three of which have been passed through a bottleneck: the query and key, passed through the same
        # bottleneck, and the residual layer to be applied in the attention self output, through another bottleneck.
        #
        # Finally, in the last case, the values for the query, key and values are the hidden states without bottleneck,
        # and the residual layer will be this value passed through a bottleneck.

        bottlenecked_hidden_states = self.bottleneck_input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bottleneck_input", None) is not None:
            with tf.name_scope(self.bottleneck_input.name):
                self.bottleneck_input.build(None)
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)


class TFFFNOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.true_hidden_size, name="dense")
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.config = config

    def call(self, hidden_states, residual_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFFFNLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        self.mobilebert_output = TFFFNOutput(config, name="output")

    def call(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.mobilebert_output(intermediate_output, hidden_states)
        return layer_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)


class TFMobileBertLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks
        self.attention = TFMobileBertAttention(config, name="attention")
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        self.mobilebert_output = TFMobileBertOutput(config, name="output")

        if self.use_bottleneck:
            self.bottleneck = TFBottleneck(config, name="bottleneck")
        if config.num_feedforward_networks > 1:
            self.ffn = [TFFFNLayer(config, name=f"ffn.{i}") for i in range(config.num_feedforward_networks - 1)]

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions,
            training=training,
        )

        attention_output = attention_outputs[0]
        s = (attention_output,)

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.mobilebert_output(intermediate_output, attention_output, hidden_states, training=training)

        outputs = (
            (layer_output,)
            + attention_outputs[1:]
            + (
                tf.constant(0),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )  # add attentions if we output them

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
        if getattr(self, "mobilebert_output", None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)
        if getattr(self, "bottleneck", None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        if getattr(self, "ffn", None) is not None:
            for layer in self.ffn:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFMobileBertEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [TFMobileBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
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


class TFMobileBertPooler(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.do_activate = config.classifier_activation
        if self.do_activate:
            self.dense = keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                activation="tanh",
                name="dense",
            )
        self.config = config

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            return first_token_tensor
        else:
            pooled_output = self.dense(first_token_tensor)
            return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFMobileBertPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)


class TFMobileBertLMPredictionHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transform = TFMobileBertPredictionHeadTransform(config, name="transform")
        self.config = config

    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        self.dense = self.add_weight(
            shape=(self.config.hidden_size - self.config.embedding_size, self.config.vocab_size),
            initializer="zeros",
            trainable=True,
            name="dense/weight",
        )
        self.decoder = self.add_weight(
            shape=(self.config.vocab_size, self.config.embedding_size),
            initializer="zeros",
            trainable=True,
            name="decoder/weight",
        )

        if self.built:
            return
        self.built = True
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    def get_output_embeddings(self):
        return self

    def set_output_embeddings(self, value):
        self.decoder = value
        self.config.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = tf.matmul(hidden_states, tf.concat([tf.transpose(self.decoder), self.dense], axis=0))
        hidden_states = hidden_states + self.bias
        return hidden_states


class TFMobileBertMLMHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFMobileBertLMPredictionHead(config, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


@keras_serializable
class TFMobileBertMainLayer(keras.layers.Layer):
    config_class = MobileBertConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        self.embeddings = TFMobileBertEmbeddings(config, name="embeddings")
        self.encoder = TFMobileBertEncoder(config, name="encoder")
        self.pooler = TFMobileBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
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
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    base_model_prefix = "mobilebert"


@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFMobileBertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: tf.Tensor | None = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


MOBILEBERT_START_DOCSTRING = r"""

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

    Parameters:
        config ([`MobileBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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


@add_start_docstrings(
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertModel(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        outputs = self.mobilebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)


@add_start_docstrings(
    """
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel, TFMobileBertPreTrainingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")
        self.seq_relationship = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    def get_lm_head(self):
        return self.predictions.predictions

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.predictions.name + "/" + self.predictions.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFMobileBertForPreTrainingOutput]:
        r"""
        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")
        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]
        ```"""
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            d_labels = {"labels": labels}
            d_labels["next_sentence_label"] = next_sentence_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TFMobileBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build(None)

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == "cls.predictions.decoder.weight":
            return tf_weight, "mobilebert.embeddings.word_embeddings.weight"
        else:
            return (tf_weight,)


@add_start_docstrings("""MobileBert Model with a `language modeling` head on top.""", MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMaskedLM(TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"seq_relationship___cls",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")

    def get_lm_head(self):
        return self.predictions.predictions

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.57,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels
        """
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        prediction_scores = self.predictions(sequence_output, training=training)

        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == "cls.predictions.decoder.weight":
            return tf_weight, "mobilebert.embeddings.word_embeddings.weight"
        else:
            return (tf_weight,)


class TFMobileBertOnlyNSPHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.seq_relationship = keras.layers.Dense(2, name="seq_relationship")
        self.config = config

    def call(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "seq_relationship", None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top.""",
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"predictions___cls", r"cls.predictions"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.cls = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        next_sentence_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFNextSentencePredictorOutput]:
        r"""
        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        ```"""
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = (
            None
            if next_sentence_label is None
            else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
        )

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return TFNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)


@add_start_docstrings(
    """
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForSequenceClassification(TFMobileBertPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForQuestionAnswering(TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFQuestionAnsweringModelOutput]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        outputs = self.mobilebert(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForTokenClassification(TFMobileBertPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "mobilebert", None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


__all__ = [
    "TFMobileBertForMaskedLM",
    "TFMobileBertForMultipleChoice",
    "TFMobileBertForNextSentencePrediction",
    "TFMobileBertForPreTraining",
    "TFMobileBertForQuestionAnswering",
    "TFMobileBertForSequenceClassification",
    "TFMobileBertForTokenClassification",
    "TFMobileBertMainLayer",
    "TFMobileBertModel",
    "TFMobileBertPreTrainedModel",
]
