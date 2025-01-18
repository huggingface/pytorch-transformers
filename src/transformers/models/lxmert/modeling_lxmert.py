# coding=utf-8
# Copyright 2018 Hao Tan, Mohit Bansal, and the HuggingFace team
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
"""PyTorch LXMERT model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_lxmert import LxmertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "unc-nlp/lxmert-base-uncased"
_CONFIG_FOR_DOC = "LxmertConfig"


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


@dataclass
class LxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    language_output: Optional[torch.FloatTensor] = None
    vision_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LxmertForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`LxmertForQuestionAnswering`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.k.
        question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    question_answering_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`LxmertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.

    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    cross_relationship_score: Optional[torch.FloatTensor] = None
    question_answering_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_lxmert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class LxmertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LxmertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LxmertAttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertSelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        # Self attention attends to itself, thus keys and queries are the same (input_tensor).
        output = self.self(
            input_tensor,
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LxmertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LxmertSelfAttentionLayer(config)
        self.intermediate = LxmertIntermediate(config)
        self.output = LxmertOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        return outputs


class LxmertXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = LxmertCrossAttentionLayer(config)

        # Self-attention Layers
        self.lang_self_att = LxmertSelfAttentionLayer(config)
        self.visn_self_att = LxmertSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = LxmertIntermediate(config)
        self.lang_output = LxmertOutput(config)
        self.visn_inter = LxmertIntermediate(config)
        self.visn_output = LxmertOutput(config)

    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.visual_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=False,
        )
        return lang_att_output, visual_att_output

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visual_inter_output = self.visn_inter(visual_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visual_output = self.visn_output(visual_inter_output, visual_input)

        return lang_output, visual_output

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        output_attentions=False,
    ):
        lang_att_output, visual_att_output = self.cross_att(
            lang_input=lang_feats,
            lang_attention_mask=lang_attention_mask,
            visual_input=visual_feats,
            visual_attention_mask=visual_attention_mask,
            output_x_attentions=output_attentions,
        )
        attention_probs = lang_att_output[1:]
        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
        )

        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        return (
            (
                lang_output,
                visual_output,
                attention_probs[0],
            )
            if output_attentions
            else (lang_output, visual_output)
        )


class LxmertVisualFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_feats, visual_pos):
        x = self.visn_fc(visual_feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(visual_pos)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = LxmertVisualFeatureEncoder(config)
        self.config = config

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_r_layers)])

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_pos,
        visual_attention_mask=None,
        output_attentions=None,
    ):
        vision_hidden_states = ()
        language_hidden_states = ()
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

        visual_feats = self.visn_fc(visual_feats, visual_pos)

        # Run language layers
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
            lang_feats = l_outputs[0]
            language_hidden_states = language_hidden_states + (lang_feats,)
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)

        # Run relational layers
        for layer_module in self.r_layers:
            v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions=output_attentions)
            visual_feats = v_outputs[0]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions=output_attentions,
            )
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
        visual_encoder_outputs = (
            vision_hidden_states,
            vision_attentions if output_attentions else None,
        )
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )
        return (
            visual_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )


class LxmertPooler(nn.Module):
    def __init__(self, config):
        super(LxmertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LxmertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(LxmertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LxmertLMPredictionHead(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertLMPredictionHead, self).__init__()
        self.transform = LxmertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            lxmert_model_embedding_weights.size(1),
            lxmert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = lxmert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(lxmert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class LxmertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class LxmertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
            }
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, self.visual_losses[key]["num"]) for key in self.visual_losses}
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class LxmertPreTrainingHeads(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertPreTrainingHeads, self).__init__()
        self.predictions = LxmertLMPredictionHead(config, lxmert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LxmertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LxmertConfig
    load_tf_weights = load_tf_weights_in_lxmert
    base_model_prefix = "lxmert"
    _supports_param_buffer_assignment = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from
    Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It's a vision and language transformer
    model, pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MSCOCO captions, and Visual
    genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss
    for question answering attribute prediction, and object tag prediction.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LxmertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LXMERT_INPUTS_DOCSTRING = r"""

    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        visual_feats (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`):
            This input represents visual features. They ROI pooled object features from bounding boxes using a
            faster-RCNN model)

            These are currently not provided by the transformers library.
        visual_pos (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`):
            This input represents spacial features corresponding to their relative (via index) visual features. The
            pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
            1.

            These are currently not provided by the transformers library.
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        visual_attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
class LxmertModel(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = LxmertEmbeddings(config)
        self.encoder = LxmertEncoder(config)
        self.pooler = LxmertPooler(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LxmertModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[LxmertModelOutput, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if visual_feats is None:
            raise ValueError("`visual_feats` cannot be `None`")
        if visual_pos is None:
            raise ValueError("`visual_pos` cannot be `None`")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=self.dtype)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * torch.finfo(self.dtype).min
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, inputs_embeds)

        # Run Lxmert encoder
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            visual_attention_mask=extended_visual_attention_mask,
            output_attentions=output_attentions,
        )

        visual_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        vision_hidden_states = visual_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]

        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            vision_attentions = visual_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (
                language_attentions,
                vision_attentions,
                cross_encoder_attentions,
            )

        hidden_states = (language_hidden_states, vision_hidden_states) if output_hidden_states else ()

        visual_output = vision_hidden_states[-1]
        lang_output = language_hidden_states[-1]
        pooled_output = self.pooler(lang_output)

        if not return_dict:
            return (lang_output, visual_output, pooled_output) + hidden_states + all_attentions

        return LxmertModelOutput(
            pooled_output=pooled_output,
            language_output=lang_output,
            vision_output=visual_output,
            language_hidden_states=language_hidden_states if output_hidden_states else None,
            vision_hidden_states=vision_hidden_states if output_hidden_states else None,
            language_attentions=language_attentions if output_attentions else None,
            vision_attentions=vision_attentions if output_attentions else None,
            cross_encoder_attentions=cross_encoder_attentions if output_attentions else None,
        )


@add_start_docstrings(
    """Lxmert Model with a specified pretraining head on top.""",
    LXMERT_START_DOCSTRING,
)
class LxmertForPreTraining(LxmertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Use of pretraining tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        # Lxmert backbone
        self.lxmert = LxmertModel(config)

        # Pre-training heads
        self.cls = LxmertPreTrainingHeads(config, self.lxmert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # Weight initialization
        # Initialize weights and apply final processing
        self.post_init()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "visual_ce": CrossEntropyLoss(reduction="none"),
            "ce": CrossEntropyLoss(),
        }

        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),
                "num": config.num_object_labels,
                "loss": "visual_ce",
            }
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),
                "num": config.num_attr_labels,
                "loss": "visual_ce",
            }
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        self.visual_losses = visual_losses

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None, mean_resizing: bool = True
    ) -> nn.Embedding:
        # Adding the following steps to resize bias to match the shape of resized embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self.cls.predictions.bias = self._resize_bias(self.cls.predictions.bias, new_num_tokens)
        return new_embeddings

    def _resize_bias(self, bias, new_num_tokens: int):
        old_num_tokens = bias.shape[0]
        if new_num_tokens <= old_num_tokens:
            new_bias = bias[:new_num_tokens]
        else:
            extra_bias = torch.zeros(new_num_tokens - old_num_tokens, device=bias.device)
            new_bias = torch.cat([bias, extra_bias])
        new_bias = nn.Parameter(new_bias)
        return new_bias

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``torch.nn.Linear``` module of the model without doing anything.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the linear layer that produces question answering logits.

        Returns:
            `nn.Module`: A torch module mapping the question answering prediction hidden states or `None` if LXMERT
            does not have a visual answering head.
        """
        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # initialize all new labels
        self._init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        obj_labels: Optional[Dict[str, Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        matched_label: Optional[torch.LongTensor] = None,
        ans: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[LxmertForPreTrainingOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        obj_labels (`Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]`, *optional*):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and
            the label score respectively
        matched_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            a one hot representation hof the correct answer *optional*

        Returns:
        """

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels`"
                " instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        lang_output, visual_output, pooled_output = (
            lxmert_output[0],
            lxmert_output[1],
            lxmert_output[2],
        )
        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]

        total_loss = (
            None
            if (labels is None and matched_label is None and obj_labels is None and ans is None)
            else torch.tensor(0.0, device=device)
        )
        if labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            total_loss += masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts["ce"](cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
        if obj_labels is not None and self.task_obj_predict:
            total_visual_loss = torch.tensor(0.0, device=input_ids.device)
            visual_prediction_scores_dict = self.obj_predict_head(visual_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info["num"]
                loss_fct_name = key_info["loss"]
                label_shape = key_info["shape"]
                weight = self.visual_loss_normalizer
                visual_loss_fct = self.loss_fcts[loss_fct_name]
                visual_prediction_scores = visual_prediction_scores_dict[key]
                visual_loss = visual_loss_fct(
                    visual_prediction_scores.view(-1, output_dim),
                    label.view(label_shape),
                )
                if visual_loss.dim() > 1:  # Regression Losses
                    visual_loss = visual_loss.mean(1)
                visual_loss = (visual_loss * mask_conf.view(-1)).mean() * weight
                total_visual_loss += visual_loss
            total_loss += total_visual_loss
        if ans is not None and self.task_qa:
            answer_loss = self.loss_fcts["ce"](answer_score.view(-1, self.num_qa_labels), ans.view(-1))
            total_loss += answer_loss

        if not return_dict:
            output = (
                lang_prediction_scores,
                cross_relationship_score,
                answer_score,
            ) + lxmert_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LxmertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=lang_prediction_scores,
            cross_relationship_score=cross_relationship_score,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )


@add_start_docstrings(
    """Lxmert Model with a visual-answering head on top for downstream QA tasks""",
    LXMERT_START_DOCSTRING,
)
class LxmertForQuestionAnswering(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Lxmert backbone
        self.lxmert = LxmertModel(config)

        self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # Weight initialization
        # Initialize weights and apply final processing
        self.post_init()

        # Loss function
        self.loss = CrossEntropyLoss()

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``torch.nn.Linear``` module of the model without doing anything.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the linear layer that produces question answering logits

        Returns:
            `nn.Module`: A torch module mapping the question answering prediction hidden states. `None`: A NoneType
            object if Lxmert does not have the visual answering head.
        """

        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # initialize all new labels
        self._init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LxmertForQuestionAnsweringOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[LxmertForQuestionAnsweringOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            A one-hot representation of the correct answer
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = lxmert_output[2]
        answer_score = self.answer_head(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.view(-1))

        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForQuestionAnsweringOutput(
            loss=loss,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )


__all__ = [
    "LxmertEncoder",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "LxmertModel",
    "LxmertPreTrainedModel",
    "LxmertVisualFeatureEncoder",
    "LxmertXLayer",
]
