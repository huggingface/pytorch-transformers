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

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
    TFForcedBOSTokenLogitsProcessor,
    TFForcedEOSTokenLogitsProcessor,
    TFForceTokensLogitsProcessor,
    TFLogitsProcessorList,
    TFMinLengthLogitsProcessor,
    TFNoBadWordsLogitsProcessor,
    TFNoRepeatNGramLogitsProcessor,
    TFRepetitionPenaltyLogitsProcessor,
    TFSuppressTokensAtBeginLogitsProcessor,
    TFSuppressTokensLogitsProcessor,
    TFTemperatureLogitsWarper,
    TFTopKLogitsWarper,
    TFTopPLogitsWarper,
)


logger = logging.get_logger(__name__)


@dataclass
class TFGreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFGreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(num_return_sequences*batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
    the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size*num_return_sequences,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_return_sequences, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this
            beam. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `tf.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    beam_indices: Optional[tf.Tensor] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this
            beam. `Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `tf.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    beam_indices: Optional[tf.Tensor] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam sample.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this
            beam. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `tf.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    beam_indices: Optional[tf.Tensor] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFBeamSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`tf.Tensor` of shape `(batch_size*num_beams, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`tf.Tensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this
            beam. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`tf.Tensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `tf.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size*num_beams, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    sequences_scores: Optional[tf.Tensor] = None
    scores: Optional[Tuple[tf.Tensor]] = None
    beam_indices: Optional[tf.Tensor] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFContrastiveSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using contrastive search.

    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


@dataclass
class TFContrastiveSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using contrastive search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(tf.Tensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(tf.Tensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: tf.Tensor = None
    scores: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = None


TFGreedySearchOutput = Union[TFGreedySearchEncoderDecoderOutput, TFGreedySearchDecoderOnlyOutput]
TFSampleOutput = Union[TFSampleEncoderDecoderOutput, TFSampleDecoderOnlyOutput]
TFBeamSearchOutput = Union[TFBeamSearchEncoderDecoderOutput, TFBeamSearchDecoderOnlyOutput]
TFBeamSampleOutput = Union[TFBeamSampleEncoderDecoderOutput, TFBeamSampleDecoderOnlyOutput]
TFContrastiveSearchOutput = Union[TFContrastiveSearchEncoderDecoderOutput, TFContrastiveSearchDecoderOnlyOutput]
TFGenerateOutput = Union[
    TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, TFContrastiveSearchOutput
]


class TFGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in [`TFPreTrainedModel`].

    The class exposes [`~generation.TFGenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.TFGenerationMixin.greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.TFGenerationMixin.contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.TFGenerationMixin.sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.TFGenerationMixin.beam_search`] if `num_beams>1`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    _seed_generator = None

    @property
    def seed_generator(self):
        warnings.warn("`seed_generator` is deprecated and will be removed in a future version.", UserWarning)
        if self._seed_generator is None:
            self._seed_generator = tf.random.Generator.from_non_deterministic_state()
        return self._seed_generator

    supports_xla_generation = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`."
        )

    def compute_transition_scores(
        self,
        sequences: tf.Tensor,
        scores: Tuple[tf.Tensor],
        beam_indices: Optional[tf.Tensor] = None,
        normalize_logits: bool = False,
    ) -> tf.Tensor:
        """
        Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
        used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

        Parameters:
            sequences (`tf.Tensor`):
                The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
                shorter if all batches finished early due to the `eos_token_id`.
            scores (`tuple(tf.Tensor)`):
                Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
                of log probabilities of tokens conditioned on log softmax of previously generated tokens Tuple of
                `tf.Tensor` with up to `max_new_tokens` elements (one element for each generated token), with each
                tensor of shape `(batch_size*num_beams, config.vocab_size)`.
            beam_indices (`tf.Tensor`, *optional*):
                Beam indices of generated token id at each generation step. `tf.Tensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.
            normalize_logits (`bool`, *optional*, defaults to `False`):
                Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

        Return:
            `tf.Tensor`: A `tf.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
                the transition scores (logits)

        Examples:

        ```python
        >>> from transformers import GPT2Tokenizer, TFAutoModelForCausalLM
        >>> import numpy as np

        >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        >>> model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer.pad_token_id = tokenizer.eos_token_id
        >>> inputs = tokenizer(["Today is"], return_tensors="tf")

        >>> # Example 1: Print the scores for each token generated with Greedy Search
        >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, normalize_logits=True
        ... )
        >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        >>> # encoder-decoder models, like BART or T5.
        >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        >>> generated_tokens = outputs.sequences[:, input_length:]
        >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
        ...     # | token | token string | logits | probability
        ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
        |   262 |  the     | -1.414 | 24.33%
        |  1110 |  day     | -2.609 | 7.36%
        |   618 |  when    | -2.010 | 13.40%
        |   356 |  we      | -1.859 | 15.58%
        |   460 |  can     | -2.508 | 8.14%

        >>> # Example 2: Reconstruct the sequence scores from Beam Search
        >>> outputs = model.generate(
        ...     **inputs,
        ...     max_new_tokens=5,
        ...     num_beams=4,
        ...     num_return_sequences=4,
        ...     return_dict_in_generate=True,
        ...     output_scores=True,
        ... )
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        ... )
        >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        >>> # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
        >>> # use case, you might want to recompute it with `normalize_logits=True`.
        >>> output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        >>> length_penalty = model.generation_config.length_penalty
        >>> reconstructed_scores = np.sum(transition_scores, axis=1) / (output_length**length_penalty)
        >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
        True
        ```"""
        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = tf.tile(tf.expand_dims(tf.range(scores[0].shape[0]), axis=1), [1, len(scores)])

        # 2. reshape scores as [batch_size, vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length
        scores = tf.transpose(tf.reshape(tf.stack(scores), (len(scores), -1)), (1, 0))
        scores = tf.reshape(scores, (-1, self.config.vocab_size, scores.shape[-1]))

        # 3. Optionally normalize the logits (across the vocab dimension)
        if normalize_logits:
            scores = tf.nn.log_softmax(scores, axis=1)

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = tf.math.reduce_max(
            tf.math.reduce_sum((1 - tf.cast(beam_indices_mask, dtype=tf.int32)), axis=-1)
        )
        beam_indices = beam_indices[:, -max_beam_length:]
        beam_indices_mask = beam_indices_mask[:, -max_beam_length:]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices = tf.where(beam_indices_mask, 0, beam_indices)

        # 6. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        token_indices = sequences[:, cut_idx:]
        gen_step_idx = tf.broadcast_to(tf.range(scores.shape[-1]), token_indices.shape)
        indices = tf.stack([beam_indices, token_indices, gen_step_idx], axis=-1)

        # 7. Compute scores
        transition_scores = tf.gather_nd(scores, indices)

        # 8. Mask out transition_scores of beams that stopped early
        transition_scores = tf.where(beam_indices_mask, 0, transition_scores)

        return transition_scores

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        if not self.can_generate():
            generate_compatible_mappings = [
                TF_MODEL_FOR_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
                TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.call).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def generate(
        self,
        inputs: Optional[tf.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        seed=None,
        **kwargs,
    ) -> Union[TFGenerateOutput, tf.Tensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate, e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`tf.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `tf.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `tf.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.TFGreedySearchDecoderOnlyOutput`],
                    - [`~generation.TFSampleDecoderOnlyOutput`],
                    - [`~generation.TFBeamSearchDecoderOnlyOutput`],
                    - [`~generation.TFBeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.TFGreedySearchEncoderDecoderOutput`],
                    - [`~generation.TFSampleEncoderDecoderOutput`],
                    - [`~generation.TFBeamSearchEncoderDecoderOutput`],
                    - [`~generation.TFBeamSampleEncoderDecoderOutput`]

        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Cast input dtypes to tf.int32 unless they're floats (which happens for some image models)
        if inputs is not None:
            if isinstance(inputs, tf.Tensor) and inputs.dtype.is_floating:
                pass
            elif isinstance(inputs, np.ndarray) and np.issubdtype(inputs.dtype, np.floating):
                pass
            else:
                inputs = tf.cast(inputs, tf.int32)
        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = tf.cast(model_kwargs["attention_mask"], tf.int32)
        if "decoder_input_ids" in model_kwargs:
            if (
                isinstance(model_kwargs["decoder_input_ids"], tf.Tensor)
                and model_kwargs["decoder_input_ids"].dtype.is_floating
            ):
                pass
            elif isinstance(model_kwargs["decoder_input_ids"], np.ndarray) and np.issubdtype(
                model_kwargs["decoder_input_ids"].dtype, np.floating
            ):
                pass
            else:
                model_kwargs["decoder_input_ids"] = tf.cast(model_kwargs["decoder_input_ids"], tf.int32)

        # 3. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask") is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            generation_config.pad_token_id = eos_token_id

        use_xla = not tf.executing_eagerly()
        if use_xla and not self.supports_xla_generation:
            raise ValueError(
                "The selected model does not support Graph mode nor XLA generation (e.g. from tf.function())"
            )

        # 4. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        # inputs_ids now has to be defined and cannot be None anymore
        batch_size = shape_list(inputs_tensor)[0]

        # 5. Prepare other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.call).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if generation_config.pad_token_id is not None and tf.math.reduce_any(
                inputs_tensor[:, -1] == generation_config.pad_token_id
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )
        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 6. Prepare model inputs which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 7. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = shape_list(input_ids)[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        # If the input length is a tensor (i.e. dynamic length), skip length checks
        if not isinstance(input_ids_seq_length, tf.Tensor):
            if (
                generation_config.min_length is not None
                and generation_config.min_length > generation_config.max_length
            ):
                raise ValueError(
                    f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger"
                    f" than the maximum length ({generation_config.max_length})"
                )
            if input_ids_seq_length >= generation_config.max_length:
                input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
                logger.warning(
                    f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                    f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                    " increasing`max_new_tokens`."
                )

        # 8. determine generation mode
        is_contrastive_search_gen_mode = (
            generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )
        is_greedy_gen_mode = (
            not is_contrastive_search_gen_mode
            and (generation_config.num_beams == 1)
            and generation_config.do_sample is False
        )
        is_beam_gen_mode = (
            not is_contrastive_search_gen_mode
            and (generation_config.num_beams > 1)
            and generation_config.do_sample is False
        )
        is_sample_gen_mode = (generation_config.num_beams == 1) and generation_config.do_sample is True
        is_beam_sample_gen_mode = (generation_config.num_beams > 1) and generation_config.do_sample is True

        # 9. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            logits_processor=logits_processor,
        )

        # 10. go into different generation modes
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                    " greedy search."
                )
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                logits_processor=logits_processor,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                **model_kwargs,
            )
        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                    " contrastive search."
                )
            # 11. run contrastive search
            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                **model_kwargs,
            )
        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config=generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                seed=seed,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_beams < generation_config.num_return_sequences:
                raise ValueError(
                    "Beam search decoding cannot return more sequences than it has beams. Please set num_beams >="
                    f" num_return_sequences, got {generation_config.num_beams} and"
                    f" {generation_config.num_return_sequences} (respectivelly)"
                )

            # 11. broadcast inputs to the desired number of beams
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                expand_in_new_axis=True,
                **model_kwargs,
            )

            # 12. run beam search
            return self.beam_search(
                input_ids,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                length_penalty=generation_config.length_penalty,
                early_stopping=generation_config.early_stopping,
                logits_processor=logits_processor,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                num_return_sequences=generation_config.num_return_sequences,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            if generation_config.num_beams < generation_config.num_return_sequences:
                raise ValueError(
                    "Beam search decoding cannot return more sequences than it has beams. Please set num_beams >="
                    f" num_return_sequences, got {generation_config.num_beams} and"
                    f" {generation_config.num_return_sequences} (respectivelly)"
                )

            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config=generation_config)

            # 12. broadcast inputs to the desired number of beams
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                expand_in_new_axis=True,
                **model_kwargs,
            )

            # 13. run beam sample (beam search with sampling)
            return self.beam_search(
                input_ids,
                do_sample=True,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                length_penalty=generation_config.length_penalty,
                early_stopping=generation_config.early_stopping,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                num_return_sequences=generation_config.num_return_sequences,
                **model_kwargs,
            )

    def _prepare_attention_mask_for_generation(
        self,
        inputs: tf.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
    ) -> tf.Tensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
        is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32)
        else:
            return tf.ones(inputs.shape[:2], dtype=tf.int32)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: tf.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder and store encoder outputs
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.call).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. vision models don't use `attention_mask`.
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        if model_input_name != self.main_input_name:  # in Keras, the first input must always be passed
            encoder_kwargs[self.main_input_name] = None
        encoder_outputs = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = encoder_outputs

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, tf.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids_start = tf.ones((batch_size, 1), dtype=tf.int32) * decoder_start_token_id

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif tf.reduce_all(decoder_input_ids[:, 0] != decoder_start_token_id):
            decoder_input_ids = tf.concat([decoder_input_ids_start, decoder_input_ids], axis=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = tf.concat(
                    (tf.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    axis=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # retrieve decoder_start_token_id for encoder-decoder models
        # fall back to bos_token_id if necessary
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[tf.Tensor] = None,
        expand_in_new_axis: bool = False,
        **model_kwargs,
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...] or [batch_size, expand_size, ...],
        depending on `expand_in_new_axis`. Beam-based approaches expect this function to be used with
        `expand_in_new_axis=True`
        """

        def _expand_tensor(tensor: tf.Tensor):
            if expand_in_new_axis:
                shape = shape_list(tensor)
                return tf.broadcast_to(tensor[:, None], (shape[0], expand_size) + tuple(shape[1:]))
            else:
                return tf.repeat(tensor, expand_size, axis=0)

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], tf.Tensor):
                    dict_to_expand[key] = _expand_tensor(dict_to_expand[key])
            return dict_to_expand

        if input_ids is not None:
            input_ids = _expand_tensor(input_ids)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _prepare_model_inputs(
        self,
        inputs: Optional[tf.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Tuple[tf.Tensor, Optional[str], Dict[str, tf.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and hasattr(self.encoder, "main_input_name")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)

        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[tf.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, tf.Tensor]] = None,
    ) -> tf.Tensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.shape[:-1]
            return tf.ones(shape, dtype=tf.int32) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, tf.Tensor):
                batch_size = value.shape[0]
                break
        return tf.ones((batch_size, 1), dtype=tf.int32) * bos_token_id

    @staticmethod
    def _extract_past_from_model_output(outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = tf.concat(
                    [attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
                )

        return model_kwargs

    def _update_model_kwargs_for_xla_generation(
        self,
        model_outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        cur_len: int,
        max_length: int,
        batch_size: int,
        is_encoder_decoder: bool = False,
        batch_axis: int = 0,
    ):
        def _initialize_attention(model_kwargs, num_padding_values, is_encoder_decoder):
            """initializes the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""
            if is_encoder_decoder:
                # One 1 for decoder_start_token_id, 0s for the currently-unfilled locations in the past_key_values tensor,
                # 1s for the actual input_ids
                decoder_attention_mask = tf.concat(
                    [
                        tf.ones((batch_size, 1), dtype=tf.int32),
                        tf.zeros((batch_size, num_padding_values), dtype=tf.int32),
                        tf.ones((batch_size, 1), dtype=tf.int32),
                    ],
                    axis=1,
                )
                mask = {"decoder_attention_mask": decoder_attention_mask}
            else:
                attention_mask = model_kwargs.pop("attention_mask")
                # 0s for the currently-unfilled locations in the past_key_values tensor, 1s for the actual input_ids
                attention_mask = tf.concat(
                    [
                        attention_mask,
                        tf.zeros((batch_size, num_padding_values), dtype=attention_mask.dtype),
                        tf.ones((batch_size, 1), dtype=attention_mask.dtype),
                    ],
                    axis=1,
                )
                mask = {"attention_mask": attention_mask}
            return mask

        def _update_attention(model_kwargs, new_past_index, is_encoder_decoder):
            """updates the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""
            update_start = tf.constant([0, 1], dtype=tf.int32) * new_past_index
            if is_encoder_decoder:
                decoder_attention_mask = model_kwargs.pop("decoder_attention_mask")
                decoder_attention_mask_update_slice = tf.ones((batch_size, 1), dtype=decoder_attention_mask.dtype)
                decoder_attention_mask = dynamic_update_slice(
                    decoder_attention_mask, decoder_attention_mask_update_slice, update_start
                )
                mask = {"decoder_attention_mask": decoder_attention_mask}
            else:
                attention_mask = model_kwargs.pop("attention_mask")
                attention_mask_update_slice = tf.ones((batch_size, 1), dtype=attention_mask.dtype)
                attention_mask = dynamic_update_slice(attention_mask, attention_mask_update_slice, update_start)
                mask = {"attention_mask": attention_mask}
            return mask

        def _initialize_past(past_key_values, num_padding_values, batch_axis):
            """initialize past_key_values with zeros -- the structure depends on `batch_axis`"""
            if batch_axis == 0:
                padding_values = tf.constant([[0, 0], [0, 0], [0, num_padding_values], [0, 0]], dtype=tf.int32)
                new_past = ()
                for past_layer in past_key_values:
                    new_past_layer = list(past_layer)
                    for i in range(len(new_past_layer[:2])):
                        new_past_layer[i] = tf.pad(past_layer[i], padding_values)
                    new_past += (tuple(new_past_layer),)
            else:
                padding_values = tf.scatter_nd(indices=[[3, 1]], updates=[num_padding_values], shape=(5, 2))
                new_past = list(past_key_values)
                for i in range(len(past_key_values)):
                    new_past[i] = tf.pad(past_key_values[i], padding_values)
            return new_past

        def _update_past(past_key_values, new_past_index, batch_axis):
            if batch_axis == 0:
                slice_start_base = tf.constant([0, 0, 1, 0])
                new_past = ()
                for past_layer in past_key_values:
                    new_past_layer = list(past_layer)
                    for i in range(len(new_past_layer[:2])):
                        update_slice = past_layer[i][:, :, -1:]
                        # Write the last slice to the first open location in the padded past_key_values array
                        # and then truncate the last slice off the array
                        new_past_layer[i] = dynamic_update_slice(
                            past_layer[i][:, :, :-1], update_slice, slice_start_base * new_past_index
                        )
                    new_past += (tuple(new_past_layer),)
            else:
                slice_start_base = tf.constant([0, 0, 0, 1, 0])
                new_past = [None for _ in range(len(past_key_values))]
                for i in range(len(past_key_values)):
                    update_slice = past_key_values[i][:, :, :, -1:]
                    # Write the last slice to the first open location in the padded past_key_values array
                    # and then truncate the last slice off the array
                    new_past[i] = dynamic_update_slice(
                        past_key_values[i][:, :, :, :-1], update_slice, slice_start_base * new_past_index
                    )
            return new_past

        past_key_values = self._extract_past_from_model_output(model_outputs)
        if past_key_values is None:
            raise ValueError(
                "No known `past_key_values variable` found in model outputs (model outputs keys:"
                f" {list(model_outputs.keys())})"
            )
        is_past_initialized = model_kwargs.pop("past_key_values", None) is not None

        if not is_past_initialized:
            # The padded version of `past_key_values` has a length of `max_length - 1`, as `past_key_values` holds information relative to
            # previous autoregressive generation steps (step 0 has no past_key_values, step 1 has 1 past_key_values value, ..., the last step
            # has `max_length - 1` past_key_values values).
            num_padding_values = max_length - cur_len - 1
            mask = _initialize_attention(model_kwargs, num_padding_values, is_encoder_decoder)
            new_past = _initialize_past(past_key_values, num_padding_values, batch_axis)
        else:
            # The new index of past_key_values to be filled corresponds to the current length of the sequence, with two
            # subtractions: -1 because past_key_values holds information regarding previous generation steps (read comment above)
            # and -1 again because in an array the index is the length of the array minus 1.
            new_past_index = cur_len - 2
            mask = _update_attention(model_kwargs, new_past_index, is_encoder_decoder)
            new_past = _update_past(past_key_values, new_past_index, batch_axis)

        # sets the updated variables (mask and past_key_values)
        model_kwargs.update(mask)
        model_kwargs["past_key_values"] = tuple(new_past)

        return model_kwargs

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsWarper`]
        instances used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = TFLogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(generation_config.eos_token_id) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TFTemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TFTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TFTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        logits_processor: Optional[TFLogitsProcessorList],
    ) -> TFLogitsProcessorList:
        """
        This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = TFLogitsProcessorList()

        # instantiate processors list
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(TFRepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(TFNoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if generation_config.bad_words_ids is not None:
            processors.append(
                TFNoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(TFMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        if generation_config.forced_bos_token_id is not None:
            processors.append(TFForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                TFForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.suppress_tokens is not None:
            processors.append(TFSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                begin_index += generation_config.forced_decoder_ids[-1][
                    0
                ]  # generation starts after the last token that is forced
            processors.append(
                TFSuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        if generation_config.forced_decoder_ids is not None:
            processors.append(TFForceTokensLogitsProcessor(generation_config.forced_decoder_ids))

        processors = self._merge_criteria_processor_list(processors, logits_processor)
        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: TFLogitsProcessorList,
        custom_list: TFLogitsProcessorList,
    ) -> TFLogitsProcessorList:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def greedy_search(
        self,
        input_ids: tf.Tensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFGreedySearchOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `call` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.TFGreedySearchDecoderOnlyOutput`], [`~generation.TFGreedySearchEncoderDecoderOutput`] or
            `tf.Tensor`: A `tf.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation.TFGreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.TFGreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForCausalLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="tf").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [
        ...         TFMinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["Today is a beautiful day, and I'm so happy to be here. I'm so happy to"]
        ```"""

        # 1. init greedy_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()

        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        use_cache = model_kwargs.pop("use_cache", self.generation_config.use_cache)
        use_xla = not tf.executing_eagerly()
        # TODO (Joao): fix cache format or find programatic way to detect cache index
        # GPT2 and other models has a slightly different cache structure, with a different batch axis
        model_name = str(self.decoder) if "EncoderDecoder" in str(self) else str(self)
        cache_batch_axis = 1 if any(model_prefix in model_name for model_prefix in ("TFGPT2", "TFCTRL")) else 0
        # some models, like XLNet, need more than the last token in the presence of past_key_values
        needs_full_input = "use_mems" in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        batch_size, cur_len = shape_list(input_ids)

        # initialize `generated` (`input_ids` padded with `pad_token_id`), `finished_sequences`
        input_ids_padding = tf.ones((batch_size, max_length - cur_len), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        # define condition fn
        def greedy_search_cond_fn(generated, finished_sequences, cur_len, model_kwargs):
            """state termination condition fn."""
            return ~tf.reduce_all(finished_sequences)

        # define condition fn
        def greedy_search_body_fn(generated, finished_sequences, cur_len, model_kwargs):
            """state update fn."""
            if model_kwargs.get("past_key_values") is None or needs_full_input:
                input_ids = generated[:, :cur_len]
            else:
                input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
            model_inputs = self.prepare_inputs_for_generation(input_ids, use_cache=use_cache, **model_kwargs)
            # forward pass to get next token logits
            model_outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = model_outputs.logits[:, -1]

            # pre-process distribution
            next_tokens_scores = logits_processor(generated, next_token_logits, cur_len)

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(next_tokens_scores)
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(model_outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.hidden_states)

            # argmax
            next_tokens = tf.argmax(next_tokens_scores, axis=-1, output_type=tf.int32)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
                next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
                next_token_is_eos = tf.math.reduce_any(
                    tf.equal(
                        tf.broadcast_to(next_tokens, (len(eos_token_id), batch_size)), tf.expand_dims(eos_token_id, -1)
                    ),
                    axis=0,
                )
                finished_sequences = finished_sequences | next_token_is_eos

            # update `generated` and `cur_len`
            update_indices = tf.stack([tf.range(batch_size), tf.broadcast_to(cur_len, [batch_size])], axis=-1)
            generated = tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)
            cur_len += 1

            # update model_kwargs
            if use_xla:
                model_kwargs = self._update_model_kwargs_for_xla_generation(
                    model_outputs=model_outputs,
                    model_kwargs=model_kwargs,
                    cur_len=cur_len,
                    max_length=max_length,
                    batch_size=batch_size,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    batch_axis=cache_batch_axis,
                )
            else:
                model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                # if we don't cache past_key_values key values we need the whole input
                if model_kwargs.get("past_key_values", None) is None:
                    # let's throw out `past_key_values` since we don't want `None` tensors
                    model_kwargs.pop("past_key_values", None)

            return generated, finished_sequences, cur_len, model_kwargs

        # 5. run generation
        # 1st generation step has to be run before to initialize `past_key_values`
        generated, finished_sequences, cur_len, model_kwargs = greedy_search_body_fn(
            generated, finished_sequences, cur_len, model_kwargs
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        maximum_iterations = max_length - cur_len
        generated, _, cur_len, _ = tf.while_loop(
            greedy_search_cond_fn,
            greedy_search_body_fn,
            (generated, finished_sequences, cur_len, model_kwargs),
            maximum_iterations=maximum_iterations,
        )

        # 6. prepare outputs
        if not use_xla:
            # cut for backward compatibility
            generated = generated[:, :cur_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights
                # and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                scores = tuple(scores) if scores is not None else None
                decoder_attentions = tuple(decoder_attentions) if decoder_attentions is not None else None
                cross_attentions = tuple(cross_attentions) if cross_attentions is not None else None
                decoder_hidden_states = tuple(decoder_hidden_states) if decoder_hidden_states is not None else None

                return TFGreedySearchEncoderDecoderOutput(
                    sequences=generated,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFGreedySearchDecoderOnlyOutput(
                    sequences=generated,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return generated

    def sample(
        self,
        input_ids: tf.Tensor,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        seed: Optional[Tuple[int, int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsWarper`]
                used to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `call` function of the model. If model is an
                encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.TFSampleDecoderOnlyOutput`], [`~generation.TFSampleEncoderDecoderOutput`] or `tf.Tensor`: A
            `tf.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation.TFSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.TFSampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForCausalLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ...     TFTopKLogitsWarper,
        ...     TFTemperatureLogitsWarper,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="tf").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [
        ...         TFMinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = TFLogitsProcessorList(
        ...     [
        ...         TFTopKLogitsWarper(50),
        ...         TFTemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> tf.random.set_seed(0)
        >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and I love my country. But when I look at Donald Trump,']
        ```"""

        # 1. init greedy_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else TFLogitsProcessorList()

        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        use_cache = model_kwargs.pop("use_cache", self.generation_config.use_cache)
        use_xla = not tf.executing_eagerly()
        # TODO (Joao): fix cache format or find programatic way to detect cache index
        # GPT2 and other models has a slightly different cache structure, with a different batch axis
        model_name = str(self.decoder) if "EncoderDecoder" in str(self) else str(self)
        cache_batch_axis = 1 if any(model_prefix in model_name for model_prefix in ("TFGPT2", "TFCTRL")) else 0
        # some models, like XLNet, need more than the last token in the presence of past_key_values
        needs_full_input = "use_mems" in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        batch_size, cur_len = shape_list(input_ids)

        # initialize `generated` (pre-populated with `pad_token_id`), `finished_sequences`
        input_ids_padding = tf.ones((batch_size, max_length - cur_len), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        def sample_cond_fn(generated, finished_sequences, cur_len, model_kwargs):
            return ~tf.reduce_all(finished_sequences)

        def sample_body_fn(generated, finished_sequences, cur_len, model_kwargs):
            if model_kwargs.get("past_key_values") is None or needs_full_input:
                input_ids = generated[:, :cur_len]
            else:
                input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
            model_inputs = self.prepare_inputs_for_generation(input_ids, use_cache=use_cache, **model_kwargs)
            # forward pass to get next token logits
            model_outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = model_outputs.logits[:, -1]

            # pre-process distribution
            next_tokens_scores = logits_processor(generated, next_token_logits, cur_len)
            next_tokens_scores = logits_warper(generated, next_tokens_scores, cur_len)

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(next_tokens_scores)
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(model_outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.hidden_states)

            # sample
            if seed is not None:
                sample_seed = seed
            else:
                sample_seed = tf.experimental.numpy.random.randint(tf.int32.min, tf.int32.max, (2,), dtype=tf.int32)
            next_tokens = tf.squeeze(
                tf.random.stateless_categorical(
                    logits=next_tokens_scores, num_samples=1, seed=sample_seed, dtype=tf.int32
                ),
                axis=1,
            )

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
                next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
                next_token_is_eos = tf.math.reduce_any(
                    tf.equal(
                        tf.broadcast_to(next_tokens, (len(eos_token_id), batch_size)), tf.expand_dims(eos_token_id, -1)
                    ),
                    axis=0,
                )
                finished_sequences = finished_sequences | next_token_is_eos

            # update `generated` and `cur_len`
            update_indices = tf.stack([tf.range(batch_size), tf.broadcast_to(cur_len, [batch_size])], axis=-1)
            generated = tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)
            cur_len += 1

            # update model_kwargs
            if use_xla:
                model_kwargs = self._update_model_kwargs_for_xla_generation(
                    model_outputs=model_outputs,
                    model_kwargs=model_kwargs,
                    cur_len=cur_len,
                    max_length=max_length,
                    batch_size=batch_size,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    batch_axis=cache_batch_axis,
                )
            else:
                model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                # if we don't cache past_key_values key values we need the whole input
                if model_kwargs.get("past_key_values", None) is None:
                    # let's throw out `past_key_values` since we don't want `None` tensors
                    model_kwargs.pop("past_key_values", None)

            return generated, finished_sequences, cur_len, model_kwargs

        # 5. run generation
        # 1st generation step has to be run before to initialize `past_key_values`
        generated, finished_sequences, cur_len, model_kwargs = sample_body_fn(
            generated, finished_sequences, cur_len, model_kwargs
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        maximum_iterations = max_length - cur_len
        generated, _, cur_len, _ = tf.while_loop(
            sample_cond_fn,
            sample_body_fn,
            (generated, finished_sequences, cur_len, model_kwargs),
            maximum_iterations=maximum_iterations,
        )

        # 6. prepare outputs
        if not use_xla:
            # cut for backward compatibility
            generated = generated[:, :cur_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights
                # and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                scores = tuple(scores) if scores is not None else None
                decoder_attentions = tuple(decoder_attentions) if decoder_attentions is not None else None
                cross_attentions = tuple(cross_attentions) if cross_attentions is not None else None
                decoder_hidden_states = tuple(decoder_hidden_states) if decoder_hidden_states is not None else None

                return TFSampleEncoderDecoderOutput(
                    sequences=generated,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFSampleDecoderOnlyOutput(
                    sequences=generated,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return generated

    @staticmethod
    def _gather_beams(nested, beam_indices, batch_axis=0):
        """Gathers the beam slices indexed by beam_indices into new beam array."""

        def gather_fn(tensor):
            if batch_axis > 0:
                # pushes all dimentions before the batch to the end, so we get (batch, beam_id, ...)
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                tensor = tf.transpose(tensor, perm=perm)

            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
            if batch_axis > 0:
                # transposes back to the original dimensions
                perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
                perm = tf.math.invert_permutation(perm)
                gathered_tensor = tf.transpose(gathered_tensor, perm=perm)

            return gathered_tensor

        return tf.nest.map_structure(gather_fn, nested)

    def beam_search(
        self,
        input_ids: tf.Tensor,
        do_sample: bool = False,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[Union[bool, str]] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        num_return_sequences: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFBeamSearchOutput, TFBeamSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search. If `do_sample` is `False`, uses
        a greedy approach, otherwise does multinomial sampling without replacement.

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent
                to the sequence length, which in turn is used to divide the score of the sequence. Since the score is
                the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences,
                while `length_penalty` < 0.0 encourages shorter sequences.
            early_stopping (`bool` or `str`, *optional*, defaults to `False`):
                Controls the stopping condition for beam-based methods, like beam-search. It accepts the following
                values: `True`, where the generation stops as soon as there are `num_beams` complete candidates;
                `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better
                candidates; `"never"`, where the beam search procedure only stops when there cannot be better
                candidates (canonical beam search algorithm).
            logits_processor (`[TFLogitsProcessorList]`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsWarper`]
                used to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `call` function of the model. If model is an
                encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.TFBeamSearchDecoderOnlyOutput`], [`~generation.TFBeamSearchEncoderDecoderOutput`] or
            `tf.Tensor`: A `tf.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation.TFBeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.TFBeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForSeq2SeqLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ... )
        >>> import tensorflow as tf

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = TFAutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="tf").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = tf.ones((1, num_beams, 1), dtype=tf.int32)
        >>> input_ids = input_ids * model.generation_config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> encoder_outputs = model.get_encoder()(encoder_input_ids, return_dict=True)
        >>> encoder_outputs.last_hidden_state = tf.repeat(
        ...     tf.expand_dims(encoder_outputs.last_hidden_state, axis=0), num_beams, axis=1
        ... )
        >>> model_kwargs = {"encoder_outputs": encoder_outputs}

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [TFMinLengthLogitsProcessor(5, eos_token_id=model.generation_config.eos_token_id)]
        ... )

        >>> outputs = model.beam_search(input_ids, logits_processor=logits_processor, **model_kwargs)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""

        def flatten_beam_dim(tensor, batch_axis=0):
            """Flattens the first two dimensions of a non-scalar array."""
            shape = shape_list(tensor)
            return tf.reshape(
                tensor,
                shape[:batch_axis] + [shape[batch_axis] * shape[batch_axis + 1]] + shape[batch_axis + 2 :],
            )

        def unflatten_beam_dim(tensor, num_beams, batch_axis=0):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            shape = shape_list(tensor)
            return tf.reshape(tensor, shape[:batch_axis] + [-1, num_beams] + shape[batch_axis + 1 :])

        # 1. init beam_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else TFLogitsProcessorList()

        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.generation_config.num_return_sequences
        )

        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        length_penalty = length_penalty if length_penalty is not None else self.generation_config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.generation_config.early_stopping

        use_cache = model_kwargs.pop("use_cache", self.generation_config.use_cache)
        use_xla = not tf.executing_eagerly()
        # TODO (Joao): fix cache format or find programatic way to detect cache index
        # GPT2 and other models has a slightly different cache structure, with a different batch axis
        model_name = str(self.decoder) if "EncoderDecoder" in str(self) else str(self)
        cache_batch_axis = 1 if any(model_prefix in model_name for model_prefix in ("TFGPT2", "TFCTRL")) else 0
        # some models, like XLNet, need more than the last token in the presence of past_key_values
        needs_full_input = "use_mems" in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        all_scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        batch_size, num_beams, cur_len = shape_list(input_ids)
        # store the prompt length of decoder
        decoder_prompt_len = cur_len

        # per batch, beam-item holding current token in loop, pre-populated with `pad_token_id`
        input_ids_padding = tf.ones((batch_size, num_beams, max_length - cur_len), dtype=tf.int32) * (
            pad_token_id or 0
        )
        running_sequences = tf.concat([input_ids, input_ids_padding], axis=-1)
        sequences = tf.ones((batch_size, num_beams, max_length), dtype=tf.int32) * (pad_token_id or 0)

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = tf.zeros((batch_size, num_beams), dtype=tf.bool)

        # per batch, beam-item score, logprobs
        running_scores = tf.tile(
            tf.expand_dims(tf.convert_to_tensor([0.0] + [-1.0e9] * (num_beams - 1)), axis=0), [batch_size, 1]
        )
        scores = tf.ones((batch_size, num_beams)) * -1.0e9

        # per batch beam indices
        running_beam_indices = tf.ones((batch_size, num_beams, max_length - decoder_prompt_len), dtype=tf.int32) * -1
        beam_indices = tf.ones((batch_size, num_beams, max_length - decoder_prompt_len), dtype=tf.int32) * -1

        # flatten beam dim
        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = flatten_beam_dim(model_kwargs["attention_mask"])

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        # define stop-condition and auto-regressive function
        def beam_search_cond_fn(
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            decoder_prompt_len,
            model_kwargs,
        ):
            """
            Beam Search termination condition function -- halts the generation loop if any of these conditions becomes
            False
            """
            # 1. is less than max length?
            not_max_length_yet = cur_len < max_length

            # 2. can the new beams still improve?
            # early_stopping == False -> apply heuristic = always get the best score from `cur_len - decoder_prompt_len`. See the discussion
            # below for more details.
            # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
            # early_stopping == "never" -> compute the best score from max_length or cur_len, depending on the sign of
            #   length_penalty. Positive length_penalty favors longer sequences, thus we use max_length there.
            if early_stopping == "never" and length_penalty > 0.0:
                best_running_score = running_scores[:, :1] / ((max_length - decoder_prompt_len) ** length_penalty)
            else:
                best_running_score = running_scores[:, :1] / (
                    tf.cast(cur_len - decoder_prompt_len, dtype=tf.float32) ** length_penalty
                )
            worst_finished_score = tf.where(
                is_sent_finished, tf.math.reduce_min(scores, axis=1, keepdims=True), -1.0e9
            )
            improvement_still_possible = tf.math.reduce_any(best_running_score > worst_finished_score)

            # 3. is there still a beam that has not finished?
            still_open_beam = ~(tf.math.reduce_all(is_sent_finished) & (early_stopping is True))

            return not_max_length_yet & still_open_beam & improvement_still_possible

        def beam_search_body_fn(
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            decoder_prompt_len,
            model_kwargs,
        ):
            """
            Beam Search iterative update function -- each iteration adds a new token and updates the best sequences
            seen so far
            """
            # 1. Forward current tokens
            if model_kwargs.get("past_key_values") is None or needs_full_input:
                input_ids = running_sequences[:, :, :cur_len]
            else:
                input_ids = tf.expand_dims(running_sequences[:, :, cur_len - 1], -1)
            model_inputs = self.prepare_inputs_for_generation(
                flatten_beam_dim(input_ids), use_cache=use_cache, **model_kwargs
            )
            model_outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits = unflatten_beam_dim(model_outputs.logits[:, -1], num_beams)

            # 2. Compute log probs
            # get log probabilities from logits, process logits with processors (*e.g.* min_length, ...), and
            # add new logprobs to existing running logprobs scores.
            log_probs = tf.nn.log_softmax(logits)
            log_probs = logits_processor(flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), cur_len)
            log_probs = unflatten_beam_dim(log_probs, num_beams)
            if do_sample:
                log_probs = logits_warper(flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), cur_len)
                log_probs = unflatten_beam_dim(log_probs, num_beams)
            log_probs_processed = log_probs
            log_probs = log_probs + tf.expand_dims(running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = tf.reshape(log_probs, (batch_size, num_beams * vocab_size))

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    all_scores.append(
                        logits_warper(
                            flatten_beam_dim(running_sequences),
                            flatten_beam_dim(log_probs_processed),
                            cur_len,
                        )
                    )
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(model_outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.hidden_states)

            # 3. Retrieve top-K
            # Each item in batch has num_beams * vocab_size candidate sequences. For each item, get the top 2*k
            # candidates with the highest log-probabilities. We gather the top 2*K beams here so that even if the
            # best K sequences reach EOS simultaneously, we have another K sequences remaining to continue the live
            # beam search.
            # Gather the top 2*K scores from _all_ beams.
            # Gather 2*k top beams.
            # Recover the beam index by floor division.
            # Recover token id by modulo division and expand Id array for broadcasting.
            # Update sequences for the 2*K top-k new sequences.
            beams_to_keep = 2 * num_beams
            if do_sample:
                topk_indices = sample_without_replacement(log_probs, beams_to_keep)
                topk_log_probs = tf.gather(log_probs, topk_indices, axis=1, batch_dims=1)
            else:
                topk_log_probs, topk_indices = tf.math.top_k(log_probs, k=beams_to_keep)
            topk_current_beam_indices = topk_indices // vocab_size
            topk_running_beam_indices = self._gather_beams(running_beam_indices, topk_current_beam_indices)
            topk_running_sequences = self._gather_beams(running_sequences, topk_current_beam_indices)
            topk_ids = topk_indices % vocab_size

            # writes the new token
            indices_batch = tf.repeat(tf.range(batch_size), [beams_to_keep])
            indices_beam = tf.tile(tf.range(beams_to_keep), [batch_size])
            update_indices = tf.stack(
                [indices_batch, indices_beam, tf.broadcast_to(cur_len, [batch_size * beams_to_keep])], axis=-1
            )
            topk_sequences = tf.tensor_scatter_nd_update(
                tensor=topk_running_sequences,
                indices=update_indices,
                updates=tf.reshape(topk_ids, [batch_size * beams_to_keep]),
            )

            # we want to store the beam indices with batch information -> real beam index = beam index % num beams
            batch_modified_indices = topk_current_beam_indices + tf.broadcast_to(
                tf.expand_dims(tf.range(batch_size) * num_beams, axis=1), topk_current_beam_indices.shape
            )
            update_indices = tf.stack(
                [
                    indices_batch,
                    indices_beam,
                    tf.broadcast_to(cur_len - decoder_prompt_len, [batch_size * beams_to_keep]),
                ],
                axis=-1,
            )
            topk_beam_indices = tf.tensor_scatter_nd_update(
                tensor=topk_running_beam_indices,
                indices=update_indices,
                updates=tf.reshape(batch_modified_indices, [batch_size * beams_to_keep]),
            )

            # 4. Check which sequences have ended
            # Update current sequences: Did the top `num_beams` sequences reach an end marker?
            # To prevent these just finished sequences from being added to the current sequences
            # set of active beam search sequences, set their log probs to a very large negative value.
            if eos_token_id is None:
                eos_in_next_token = tf.zeros(topk_sequences[:, :, cur_len].shape, dtype=tf.bool)
            else:
                eos_in_next_token = tf.math.reduce_any(
                    tf.equal(
                        tf.broadcast_to(
                            topk_sequences[:, :, cur_len],
                            [len(eos_token_id)] + topk_sequences[:, :, cur_len].shape,
                        ),
                        tf.expand_dims(tf.expand_dims(eos_token_id, -1), -1),
                    ),
                    axis=0,
                )
            did_topk_just_finished = eos_in_next_token & tf.broadcast_to(
                tf.concat((tf.ones((num_beams), dtype=tf.bool), tf.zeros((num_beams), dtype=tf.bool)), axis=0),
                shape_list(eos_in_next_token),
            )

            # non-top `num_beams` eos tokens can't be used to finish a beam, but the others can't be used in the next
            # running sentences either
            running_topk_log_probs = topk_log_probs + tf.cast(eos_in_next_token, tf.float32) * -1.0e9

            # 5. Get running sequences scores for next
            # Determine the top k beam indices (from top 2*k beams) from log probs and gather top k beams
            # (from top 2*k beams).
            next_topk_indices = tf.math.top_k(running_topk_log_probs, k=num_beams)[1]
            next_running_sequences, next_running_scores, next_running_beam_indices = self._gather_beams(
                [topk_sequences, running_topk_log_probs, topk_beam_indices], next_topk_indices
            )

            # 6. Process topk logits
            # Further process log probs:
            # - add length penalty
            # - make sure no scores can be added anymore if beam is full
            # - make sure still running sequences cannot be chosen as finalized beam
            topk_log_probs = topk_log_probs / (
                tf.cast(cur_len + 1 - decoder_prompt_len, dtype=tf.float32) ** length_penalty
            )
            beams_in_batch_are_full = tf.broadcast_to(
                tf.math.reduce_all(is_sent_finished, axis=-1, keepdims=True), shape_list(did_topk_just_finished)
            ) & (early_stopping is True)
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += tf.cast(add_penalty, tf.float32) * -1.0e9

            # 7. Get scores, sequences, is sentence finished for next.
            # Combine sequences, scores, and flags along the beam dimension and compare new finished sequence scores
            # to existing finished scores and select the best from the new set of beams
            merged_sequences = tf.concat([sequences, topk_sequences], axis=1)
            merged_scores = tf.concat([scores, topk_log_probs], axis=1)
            merged_beams = tf.concat([beam_indices, topk_beam_indices], axis=1)
            merged_is_sent_finished = tf.concat([is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = tf.math.top_k(merged_scores, k=num_beams)[1]
            next_sequences, next_scores, next_beam_indices, next_is_sent_finished = self._gather_beams(
                [merged_sequences, merged_scores, merged_beams, merged_is_sent_finished], topk_merged_indices
            )

            # 8. Prepare data for the next iteration
            # Determine the top k beam indices from the original set of all beams. With these, gather the top k
            # beam-associated caches.
            cur_len = cur_len + 1
            if "past_key_values" in model_outputs:
                cache = tf.nest.map_structure(
                    lambda tensor: unflatten_beam_dim(tensor, num_beams, batch_axis=cache_batch_axis),
                    model_outputs.past_key_values,
                )
                next_running_indices = self._gather_beams(topk_current_beam_indices, next_topk_indices)
                next_cache = self._gather_beams(cache, next_running_indices, batch_axis=cache_batch_axis)
                model_outputs["past_key_values"] = tf.nest.map_structure(
                    lambda tensor: flatten_beam_dim(tensor, batch_axis=cache_batch_axis), next_cache
                )

            if use_xla:
                next_model_kwargs = self._update_model_kwargs_for_xla_generation(
                    model_outputs=model_outputs,
                    model_kwargs=model_kwargs,
                    cur_len=cur_len,
                    max_length=max_length,
                    batch_size=(batch_size * num_beams),
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    batch_axis=cache_batch_axis,
                )
            else:
                next_model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # if we don't cache past_key_values key values we need the whole input
                if model_kwargs.get("past_key_values", None) is None:
                    # let's throw out `past_key_values` since we don't want `None` tensors
                    model_kwargs.pop("past_key_values", None)

            return (
                cur_len,
                next_running_sequences,
                next_running_scores,
                next_running_beam_indices,
                next_sequences,
                next_scores,
                next_beam_indices,
                next_is_sent_finished,
                decoder_prompt_len,
                next_model_kwargs,
            )

        # 5. run generation
        # 1st generation step has to be run before to initialize `past_key_values` (if active)
        (
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            decoder_prompt_len,
            model_kwargs,
        ) = beam_search_body_fn(
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            decoder_prompt_len,
            model_kwargs,
        )

        # 2-to-n generation steps can then be run in autoregressive fashion (only in case 1st generation step does
        # NOT yield EOS token though)
        maximum_iterations = max_length - cur_len
        (
            cur_len,
            running_sequences,
            running_scores,
            running_beam_indices,
            sequences,
            scores,
            beam_indices,
            is_sent_finished,
            decoder_prompt_len,
            _,
        ) = tf.while_loop(
            beam_search_cond_fn,
            beam_search_body_fn,
            (
                cur_len,
                running_sequences,
                running_scores,
                running_beam_indices,
                sequences,
                scores,
                beam_indices,
                is_sent_finished,
                decoder_prompt_len,
                model_kwargs,
            ),
            maximum_iterations=maximum_iterations,
        )

        # 6. prepare outputs
        # Account for the edge-case where there are no finished sequences for a particular batch item. If so, return
        # running sequences for that batch item.
        none_finished = tf.math.reduce_any(is_sent_finished, axis=1)
        sequences = tf.where(none_finished[:, None, None], sequences, running_sequences)
        beam_indices = tf.where(none_finished[:, None, None], beam_indices, running_beam_indices)

        # Apply the length penalty so that running scores match the finalized scores if they are used
        running_scores = running_scores / (tf.cast(cur_len - decoder_prompt_len, dtype=tf.float32) ** length_penalty)
        scores = tf.where(none_finished[:, None], scores, running_scores)

        # Take best beams for each batch (the score is sorted in descending order)
        sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
        scores = flatten_beam_dim(scores[:, :num_return_sequences])
        beam_indices = flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

        if not use_xla:
            # Cut for backward compatibility
            sequences = sequences[:, :cur_len]
            beam_indices = beam_indices[:, : cur_len - decoder_prompt_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                output_cls = TFBeamSampleEncoderDecoderOutput if do_sample else TFBeamSearchEncoderDecoderOutput
                return output_cls(
                    sequences=sequences,
                    sequences_scores=scores,
                    scores=all_scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                output_cls = TFBeamSampleDecoderOnlyOutput if do_sample else TFBeamSearchDecoderOnlyOutput
                return output_cls(
                    sequences=sequences,
                    sequences_scores=scores,
                    scores=all_scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequences

    def contrastive_search(
        self,
        input_ids: tf.Tensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        logits_warper: Optional[TFLogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[TFContrastiveSearchOutput, tf.Tensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            top_k (`int`, *optional*, defaults to 1):
                The size of the candidate set that is used to re-rank for contrastive search
            penalty_alpha (`float`, *optional*, defaults to 0):
                The degeneration penalty for contrastive search; activate when it is larger than 0
            logits_processor (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsWarper`]
                used to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `call` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`~generation.TFContrastiveSearchDecoderOnlyOutput`],
            [`~generation.TFContrastiveSearchEncoderDecoderOutput`] or `tf.Tensor`: A `tf.Tensor` containing the
            generated tokens (default behaviour) or a [`~generation.TFContrastiveySearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation.TFContrastiveSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, TFAutoModelForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> model = TFAutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> # set pad_token_id to eos_token_id because OPT does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> input_prompt = "DeepMind Company is"
        >>> input_ids = tokenizer(input_prompt, return_tensors="tf")
        >>> outputs = model.contrastive_search(**input_ids, penalty_alpha=0.6, top_k=4, max_length=64)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.\n\nIn this post, we talk about the benefits of deep learning in business and how it']
        ```"""

        def gather_best_candidate(nested, selected_idx_stacked, batch_axis=0):
            """Gathers the slices indexed by selected_idx_stacked from a potentially nested structure of tensors."""

            def gather_fn(tensor):
                gathered_tensor = tf.gather(params=tensor, indices=selected_idx_stacked, axis=batch_axis)
                return gathered_tensor

            return tf.nest.map_structure(gather_fn, nested)

        # 1. init greedy_search values
        logits_processor = logits_processor if logits_processor is not None else TFLogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else TFLogitsProcessorList()
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        use_cache = True  # In contrastive search, we always use cache
        model_kwargs.pop("use_cache", None)

        use_xla = not tf.executing_eagerly()
        # TODO (Joao): fix cache format or find programatic way to detect cache index
        # GPT2 and other models has a slightly different cache structure, with a different batch axis
        model_name = str(self.decoder) if "EncoderDecoder" in str(self) else str(self)
        cache_batch_axis = 1 if any(model_prefix in model_name for model_prefix in ("TFGPT2", "TFCTRL")) else 0

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
        cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

        # 3. init tensors to use for "xla-compileable" generate function
        batch_size, cur_len = shape_list(input_ids)

        # initialize `generated` (`input_ids` padded with `pad_token_id`), `finished_sequences`
        input_ids_padding = tf.ones((batch_size, max_length - cur_len), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        # define condition fn
        def contrastive_search_cond_fn(
            generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables
        ):
            """state termination condition fn."""
            return ~tf.reduce_all(finished_sequences)

        # define condition fn
        def contrastive_search_body_fn(
            generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables
        ):
            """state update fn."""

            # if the first step in the loop, encode all the prefix and obtain: (1) past_key_values;
            # (2) last_hidden_states; (3) logit_for_next_step; (4) update model kwargs for the next step
            if model_kwargs.get("past_key_values") is None:
                # prepare inputs
                model_inputs = self.prepare_inputs_for_generation(
                    generated[:, :cur_len], use_cache=use_cache, **model_kwargs
                )

                # encode the given prefix and prepare model inputs; encoder-decoder model process the prefix and save
                # the `encoder_outputs`
                outputs = self(
                    **model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions
                )

                # last decoder hidden states will be used to compute the degeneration penalty (cosine similarity with
                # previous tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # XLA: last_hidden_states normally grows at each step, but in XLA it is padded so as to be used across
                # iterations (with fixed shapes)
                if use_xla:
                    last_hidden_states = tf.pad(last_hidden_states, [[0, 0], [0, max_length - cur_len], [0, 0]])

                # next logit for contrastive search to select top-k candidate tokens
                logit_for_next_step = outputs.logits[:, -1, :]

                if use_xla:
                    model_kwargs = self._update_model_kwargs_for_xla_generation(
                        model_outputs=outputs,
                        model_kwargs=model_kwargs,
                        cur_len=cur_len,
                        max_length=max_length,
                        batch_size=batch_size,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                        batch_axis=cache_batch_axis,
                    )
                else:
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                # Expands model inputs top_k times, for batched forward passes (akin to beam search).
                _, model_kwargs = self._expand_inputs_for_generation(
                    expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
                )

                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    not isinstance(past_key_values[0], (tuple, tf.Tensor))
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )
            else:
                logit_for_next_step = next_step_cached_variables["logit_for_next_step"]
                last_hidden_states = next_step_cached_variables["last_hidden_states"]
                outputs = next_step_cached_variables["outputs"]

            # contrastive_search main logic start:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall; (2) candidate re-rank by
            # degeneration penalty

            logit_for_next_step = logits_processor(generated, logit_for_next_step, cur_len)
            logit_for_next_step = logits_warper(generated, logit_for_next_step, cur_len)
            next_probs = stable_softmax(logit_for_next_step, axis=-1)
            top_k_probs, top_k_ids = tf.math.top_k(next_probs, k=top_k)

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(logit_for_next_step)
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(outputs.decoder_attentions)
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(outputs.attentions)
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(outputs.cross_attentions)

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(outputs.decoder_hidden_states)
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(outputs.hidden_states)

            # Replicates the new past_key_values to match the `top_k` candidates
            model_kwargs["past_key_values"] = tf.nest.map_structure(
                lambda tensor: tf.repeat(tensor, top_k, axis=cache_batch_axis), model_kwargs["past_key_values"]
            )

            # compute the candidate tokens by the language model and collects their hidden_states
            next_model_inputs = self.prepare_inputs_for_generation(
                tf.reshape(top_k_ids, [-1, 1]), use_cache=use_cache, **model_kwargs
            )
            outputs = self(
                **next_model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions
            )
            next_past_key_values = self._extract_past_from_model_output(outputs)

            logits = outputs.logits[:, -1, :]
            # name is different for encoder-decoder and decoder-only models
            if self.config.is_encoder_decoder:
                next_hidden = outputs.decoder_hidden_states[-1]
                full_hidden_states = outputs.decoder_hidden_states
            else:
                next_hidden = outputs.hidden_states[-1]
                full_hidden_states = outputs.hidden_states
            context_hidden = tf.repeat(last_hidden_states[:, :cur_len, :], top_k, axis=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence
            selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)

            # converts indices to a dimension of top_k to the stacked top_k * batch_size dimension, for indexing
            # without a need to reshape on tensors that have these two dimensions stacked
            selected_idx_stacked = selected_idx + tf.range(selected_idx.shape[0], dtype=tf.int64) * top_k

            # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing
            # the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores
            # (model confidence minus degeneration penalty); (6) decoder hidden_states
            next_tokens = tf.gather(top_k_ids, selected_idx, axis=1, batch_dims=1)
            next_hidden = gather_best_candidate(next_hidden, selected_idx_stacked)

            # XLA: last_hidden_states normally grows at each step, but in XLA it is padded so as to be used across
            # iterations (with fixed shapes)
            if use_xla:
                last_hidden_states = dynamic_update_slice(last_hidden_states, next_hidden, [0, cur_len, 0])
            else:
                last_hidden_states = tf.concat([last_hidden_states, next_hidden], axis=1)

            next_decoder_hidden_states = gather_best_candidate(full_hidden_states, selected_idx_stacked)
            next_past_key_values = gather_best_candidate(
                next_past_key_values, selected_idx_stacked, batch_axis=cache_batch_axis
            )
            logit_for_next_step = gather_best_candidate(logits, selected_idx_stacked)

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration
            if self.config.is_encoder_decoder:
                next_step_cross_attentions = ()
                next_step_decoder_attentions = ()
                if output_attentions:
                    next_step_cross_attentions = gather_best_candidate(outputs.cross_attentions, selected_idx_stacked)
                    next_step_decoder_attentions = gather_best_candidate(
                        outputs.decoder_attentions, selected_idx_stacked
                    )
                outputs = TFSeq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                    decoder_attentions=next_step_decoder_attentions or None,
                    cross_attentions=next_step_cross_attentions or None,
                )
            else:
                next_step_attentions = ()
                if output_attentions:
                    next_step_attentions = gather_best_candidate(outputs.attentions, selected_idx_stacked)
                outputs = TFCausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                    attentions=next_step_attentions or None,
                )
            # contrastive_search main logic end

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
                next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
                next_token_is_eos = tf.math.reduce_any(
                    tf.equal(
                        tf.broadcast_to(next_tokens, (len(eos_token_id), batch_size)), tf.expand_dims(eos_token_id, -1)
                    ),
                    axis=0,
                )
                finished_sequences = finished_sequences | next_token_is_eos

            # update `generated` and `cur_len`
            update_indices = tf.stack([tf.range(batch_size), tf.broadcast_to(cur_len, [batch_size])], axis=-1)
            generated = tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)
            cur_len += 1

            if use_xla:
                # NOTE: 1) relative to other generation strategies, contrastive search is always running forward
                # passes one step ahead -- hence the `cur_len=cur_len + 1`; 2) the attention mask here is expanded from
                # [batch_size, ...] to [batch_size*top_k, ...] -- hence the `batch_size=batch_size * top_k`
                model_kwargs = self._update_model_kwargs_for_xla_generation(
                    model_outputs=outputs,
                    model_kwargs=model_kwargs,
                    cur_len=cur_len + 1,
                    max_length=max_length,
                    batch_size=batch_size * top_k,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    batch_axis=cache_batch_axis,
                )
            else:
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

            next_step_cached_variables = {
                "logit_for_next_step": logit_for_next_step,
                "last_hidden_states": last_hidden_states,
                "outputs": outputs,
            }
            return generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables

        # 5. run generation
        # 1st generation step has to be run before to initialize `past_key_values`
        generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables = contrastive_search_body_fn(
            generated, finished_sequences, cur_len, model_kwargs, None
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        maximum_iterations = max_length - cur_len
        generated, _, cur_len, _, _ = tf.while_loop(
            contrastive_search_cond_fn,
            contrastive_search_body_fn,
            (generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables),
            maximum_iterations=maximum_iterations,
        )

        # 6. prepare outputs
        if not use_xla:
            # cut for backward compatibility
            generated = generated[:, :cur_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # if model is an encoder-decoder, retrieve encoder attention weights
                # and hidden states
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

                scores = tuple(scores) if scores is not None else None
                decoder_attentions = tuple(decoder_attentions) if decoder_attentions is not None else None
                cross_attentions = tuple(cross_attentions) if cross_attentions is not None else None
                decoder_hidden_states = tuple(decoder_hidden_states) if decoder_hidden_states is not None else None

                return TFContrastiveSearchEncoderDecoderOutput(
                    sequences=generated,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFContrastiveSearchDecoderOnlyOutput(
                    sequences=generated,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return generated


def scatter_values_on_batch_indices(values, batch_indices):
    shape = shape_list(batch_indices)
    # broadcast batch dim to shape
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)


def sample_without_replacement(logits, num_samples):
    """
    categorical sampling without replacement is currently not implemented the gumbel-max trick will do for now see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(shape_list(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return indices


def _ranking_fast(
    context_hidden: tf.Tensor,
    next_hidden: tf.Tensor,
    next_top_k_probs: tf.Tensor,
    alpha: float,
    beam_width: int,
) -> tf.Tensor:
    """
    Reranks the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens), as described
    in the paper "A Contrastive Framework for Neural Text Generation". Returns the index of the best candidate for each
    row in the batch.
    """
    norm_context_hidden = context_hidden / tf.norm(context_hidden, axis=2, keepdims=True)
    norm_next_hidden = next_hidden / tf.norm(next_hidden, axis=2, keepdims=True)
    cosine_matrix = tf.squeeze(tf.linalg.matmul(norm_context_hidden, norm_next_hidden, transpose_b=True), axis=-1)
    degeneration_penalty = tf.reduce_max(cosine_matrix, axis=-1)
    next_top_k_probs = tf.reshape(next_top_k_probs, shape=[-1])
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = tf.reshape(contrastive_score, shape=[-1, beam_width])
    selected_idx = tf.argmax(contrastive_score, axis=1)
    return selected_idx
