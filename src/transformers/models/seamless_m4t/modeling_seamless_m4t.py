# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch SeamlessM4T model."""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_seamless_m4t import SeamlessM4TConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/hf-seamless-m4t-medium"
_CONFIG_FOR_DOC = "SeamlessM4TConfig"


@dataclass
class SeamlessM4TGenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4TModel`], [`SeamlessM4TForTextToText`],
    [`SeamlessM4TForTextToSpeech`], [`SeamlessM4TForSpeechToSpeech`] and [`SeamlessM4TForTextToSpeech`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
    """

    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None


SEAMLESS_M4T_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
    """

SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
        """

SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART = r"""
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

M4T_MODEL_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

M4T_TEXT_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

M4T_SPEECH_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART


############ UTILS ################


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


def format_speech_generation_kwargs(kwargs):
    """
    Format kwargs for SeamlessM4T models that generate speech, attribute kwargs to either the text generation or the
    speech generation models.

    Args:
        kwargs (`dict`)`:
             Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for one generation but not for the
                other.
    """
    # attribute kwargs to models
    kwargs_text = {}
    kwargs_speech = {}
    for key, value in kwargs.items():
        if key.startswith("text_"):
            key = key[len("text_") :]
            kwargs_text[key] = value
        elif key.startswith("speech_"):
            key = key[len("speech_") :]
            kwargs_speech[key] = value
        elif key == "generation_config":
            kwargs_text[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return kwargs_text, kwargs_speech


############ SPEECH ENCODER related code ################


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->SeamlessM4TConformer, feat_extract_activation->speech_encoder_hidden_act
class SeamlessM4TConformerPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            if hasattr(self.conv, "parametrizations"):
                weight_g = self.conv.parametrizations.weight.original0
                weight_v = self.conv.parametrizations.weight.original1
            else:
                weight_g = self.conv.weight_g
                weight_v = self.conv.weight_v
            deepspeed.zero.register_external_parameter(self, weight_v)
            deepspeed.zero.register_external_parameter(self, weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = SeamlessM4TConformerSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRotaryPositionalEmbedding with Wav2Vec2->SeamlessM4T, num_attention_heads->speech_encoder_attention_heads
class SeamlessM4TConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.speech_encoder_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        # Embeddings are computed in the dtype of the inv_freq constant
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # Computed embeddings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        return self.cached_rotary_positional_embedding


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding with Wav2Vec2->SeamlessM4T
class SeamlessM4TConformerRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module."""

    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        # Reset the positional encodings
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` is the position of query vector and `j` is the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.int64).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reverse the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        self.extend_pe(hidden_states)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSamePadLayer with Wav2Vec2->SeamlessM4T
class SeamlessM4TConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class SeamlessM4TConformerFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SeamlessM4TConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        self.intermediate_dropout = nn.Dropout(dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SeamlessM4TConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding="same",
            groups=config.hidden_size,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SeamlessM4TConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4TConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        super().__init__()

        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        self.num_heads = config.speech_encoder_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        if self.position_embeddings_type == "relative":
            # linear transformation for positional encoding
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                    " 'relative'"
                )
            # apply relative_position_embeddings to qk scores
            # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        # apply attention_mask if necessary
        if attention_mask is not None:
            scores = scores + attention_mask

        # => (batch, head, time1, time2)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # => (batch, head, time1, d_k)
        hidden_states = torch.matmul(probs, value)

        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        return hidden_states, probs

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_rotary_embedding
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_relative_embeddings
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. project positional embeddings
        # => (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        # 2. Add bias to query
        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # => (batch, head, time1, time2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. then compute matrix b and matrix d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        # 6. sum matrices
        # => (batch, head, time1, time2)
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class SeamlessM4TConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4T, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4TConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4TConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4TConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = SeamlessM4TConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weigts = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weigts


class SeamlessM4TConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.position_embeddings_type == "relative":
            self.embed_positions = SeamlessM4TConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = SeamlessM4TConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList(
            [SeamlessM4TConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = (
                True if self.training and (dropout_probability < self.config.speech_encoder_layerdrop) else False
            )
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        relative_position_embeddings,
                        output_attentions,
                        conv_attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SeamlessM4TConformerAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout

        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.activation = nn.GLU(dim=1)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn = SeamlessM4TConformerSelfAttention(config, use_position_embeddings=False)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply pooling before feeding to the multihead-attention layer.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states


class SeamlessM4TConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(SeamlessM4TConformerAdapterLayer(config) for _ in range(config.num_adapter_layers))

    def forward(self, hidden_states, attention_mask):
        # down project hidden_states if necessary

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ TEXT / UNITS related code ################


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100ScaledWordEmbedding with M2M100->SeamlessM4T
class SeamlessM4TScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding
class SeamlessM4TSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length


class SeamlessM4TAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.bart.modeling_bart.BartAttention.__init__ with Bart->SeamlessM4T
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SeamlessM4TConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if encoder_hidden_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = encoder_hidden_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == encoder_hidden_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `encoder_hidden_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == encoder_hidden_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(encoder_hidden_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeDenseActDense with NllbMoe->SeamlessM4T,DenseActDense->FeedForwardNetwork, d_model->hidden_size
class SeamlessM4TFeedForwardNetwork(nn.Module):
    def __init__(self, config: SeamlessM4TConfig, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SeamlessM4TEncoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4TConfig, encoder_ffn_dim=None, encoder_attention_heads=None):
        super().__init__()
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SeamlessM4TDecoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4TConfig, decoder_ffn_dim=None, decoder_attention_heads=None):
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(config.dropout)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = SeamlessM4TAttention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`):
                encoder attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by
                very large negative values.
            past_key_value (`Tuple(torch.FloatTensor)`):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value += cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


############ SUB-MODELS related code ################


class SeamlessM4TPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SeamlessM4TConfig
    base_model_prefix = "seamless_m4t"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SeamlessM4TEncoderLayer", "SeamlessM4TDecoderLayer", "SeamlessM4TConformerEncoderLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SeamlessM4TConformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, SeamlessM4TConformerPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, SeamlessM4TConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        pad = kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.floor()

    def compute_last_hidden_states_per_sample(
        self,
        hidden_states: Tuple[Tuple[torch.Tensor]],
        beam_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the last hidden states.

        Parameters:
            hidden_states (`Tuple[Tuple[torch.Tensor]]`):
                The generated hidden states. Tuple (one element for each generated token) of tuples (one element for
                each layer of the decoder) of torch.FloatTensor of shape (batch_size*num_beams*num_return_sequences,
                generated_length, hidden_size).
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.

        Return:
            `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length, hidden_size)`
            containing
                the last hidden states.
        ```"""
        # 1. First, let's compute last_hidden_states from hidden_states.
        # For each generation step, takes the hidden state from the last layer.
        # shape: (batch_size*vocab_size*num_return_sequences, # generation_steps, hidden_dim)
        last_hidden_states = torch.concat([hidden_states[-1] for hidden_states in hidden_states], dim=1)

        # 2. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        # in that case, return directly last_hidden_states
        if beam_indices is None:
            return last_hidden_states

        # 3. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 4. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards anyways
        beam_indices[beam_indices_mask] = 0

        # 5. expand beam_indices to last_hidden_states dim
        beam_indices = beam_indices.unsqueeze(-1)
        beam_indices = beam_indices.expand(-1, -1, last_hidden_states.shape[-1])

        # 6. select the right candidate for each beam
        # in other words, new_last_hidden_states[i,j,k] = last_hidden_states[beam_indices[i,j,k], j, k] for all i, j, k
        last_hidden_states = torch.gather(last_hidden_states, 0, beam_indices)

        return last_hidden_states


@add_start_docstrings(
    """Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.
    Each layer is a [`SeamlessM4TConformerEncoderLayer`].""",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TSpeechEncoder(SeamlessM4TPreTrainedModel):
    main_input_name = "input_features"

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        self.feature_projection = SeamlessM4TConformerFeatureProjection(config)
        self.encoder = SeamlessM4TConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=0.0)
        self.adapter = SeamlessM4TConformerAdapter(config) if config.add_adapter else None
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4TSpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        hidden_states = self.feature_projection(input_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        hidden_states = self.inner_layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# inspired from MBart and NllbMoe
@add_start_docstrings(
    "Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4TEncoderLayer`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
        is_t2u_encoder (`bool`, *optional*, defaults to `False`):
            indicates if it belongs to the text-to-units model, in which case it won't have input embeddings
    """,
)
class SeamlessM4TEncoder(SeamlessM4TPreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size

        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings

        if not self.is_t2u_encoder:
            embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            self.embed_tokens = SeamlessM4TScaledWordEmbedding(
                config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
            )

            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight

            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4TEncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and self.is_t2u_encoder:
            raise ValueError(
                "You cannot pass input_ids to the encoder of the text_to_units model. Pass inputs_embeds instead."
            )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.is_t2u_encoder:
            embed_pos = self.embed_positions(input)

            hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.device)
        else:
            hidden_states = inputs_embeds

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.forward,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


@add_start_docstrings(
    "Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4TDecoderLayer`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
    """,
)
class SeamlessM4TDecoder(SeamlessM4TPreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # if embed_tokens defined, use its shape instead
            self.embed_tokens = SeamlessM4TScaledWordEmbedding(
                embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx, embed_scale=embed_scale
            )
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = SeamlessM4TScaledWordEmbedding(
                self.vocab_size, config.hidden_size, self.padding_idx, embed_scale=embed_scale
            )

        self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4TDecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[3],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4TEncoder`] without embeddings and the decoder is a [`SeamlessM4TDecoder`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
class SeamlessM4TTextToUnitModel(SeamlessM4TPreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)

        self.encoder = SeamlessM4TEncoder(config, is_t2u_encoder=True)
        self.decoder = SeamlessM4TDecoder(config, embed_tokens_decoder)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4TTextToUnit`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
class SeamlessM4TTextToUnitForConditionalGeneration(SeamlessM4TPreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        # update config - used principaly for bos_token_id etc.
        config = copy.deepcopy(config)
        for param, val in config.to_dict().items():
            if param.startswith("t2u_"):
                config.__setattr__(param[4:], val)
        super().__init__(config)

        self.model = SeamlessM4TTextToUnitModel(config, embed_tokens_decoder)

        self.lm_head = nn.Linear(config.hidden_size, config.t2u_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def _tie_weights(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())


############ VOCODER related code ################


HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SeamlessM4TConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        for layer in self.convs1:
            weight_norm(layer)
        for layer in self.convs2:
            weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class SeamlessM4TVariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout

        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=1,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Input: B x T x C; Output: B x T
        hidden_states = self.conv1(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        hidden_states = self.conv2(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(dim=2)


class SeamlessM4THifiGan(nn.Module):
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__()
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                model_in_dim)`, or un-batched and of shape `(sequence_length, model_in_dim)`. Note that `model_in_dim`
                is the sum of `config.unit_embed_dim`, `config.lang_embed_dim` and `config.spkr_embed_dim`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """

        hidden_states = self.conv_pre(input_embeds)
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        # remove seq-len dim since this collapses to 1
        waveform = hidden_states.squeeze(1)

        return waveform


@add_start_docstrings(
    """Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).""",
    HIFIGAN_START_DOCSTRING,
)
class SeamlessM4TCodeHifiGan(PreTrainedModel):
    config_class = SeamlessM4TConfig
    main_input_name = "input_embeds"
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)

        self.pad_token_id = config.t2u_pad_token_id
        self.dur_predictor = SeamlessM4TVariancePredictor(config)

        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        self.hifi_gan = SeamlessM4THifiGan(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        unit_lengths = (input_ids != self.pad_token_id).sum(1)

        # take care of edge cases where no padding or too many padding
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)

        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()

        return unit_lengths

    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (
                torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1
            )

        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

        # conv_pre
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        # upsampler
        for i, (upsample_rate, kernel_size) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)
        ):
            input_lengths = _transpose_conv_out_length(
                input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
            )

        # resblock
        for i in range(len(self.config.upsample_rates)):
            for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(
                        input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                    )

                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

        # conv_post
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        return input_lengths

    def forward(
        self, input_ids: torch.LongTensor, spkr_id: torch.Tensor, lang_id: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            spkr_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        spkr = self.speaker_embedding(spkr_id).transpose(1, 2)
        lang = self.language_embedding(lang_id).transpose(1, 2)

        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
        # B x C x T
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            # if batched sample, need to interleave per sample, and pad -> loss of parallelism
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]

            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)

        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)

        hidden_states = self.hifi_gan(hidden_states)

        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        weight_norm(self.hifi_gan.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)


############ WHOLE MODEL related code ################


@add_start_docstrings(
    "The text-to-text SeamlessM4T Model transformer which can be used for T2TT.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForTextToText(SeamlessM4TPreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.text_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = attention_mask

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_ids=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.Tensor` of varying shape depending on the modality, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
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
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
            [`~utils.ModelOutput`] types are:
                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # prepare text_decoder_input_ids
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            batch_size = len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds"))

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                # also accept __xxx__
                tgt_lang = tgt_lang.replace("__", "")
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in
                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                    )
                # tgt_lang gets priority over decoder input ids
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            logger.warning(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )

        return super().generate(
            input_ids,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    "The speech-to-text SeamlessM4T Model transformer which can be used for S2TT.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForSpeechToText(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.speech_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                encoder_outputs[0].device
            )
            encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_features=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.

            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
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
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
            [`~utils.ModelOutput`] types are:
                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        input_features = input_features if input_features is not None else kwargs.pop("inputs")
        if tgt_lang is not None:
            inputs = kwargs.get("input_embeds") if input_features is None else input_features
            inputs = (
                inputs
                if inputs is not None
                else kwargs.get("encoder_outputs", {"last_hidden_state": None})["last_hidden_state"]
            )
            batch_size = len(inputs)

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                # also accept __xxx__
                tgt_lang = tgt_lang.replace("__", "")
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in
                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                    )
                # tgt_lang gets priority over decoder input ids
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            logger.warning(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )
        return super().generate(
            input_features,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    "The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForTextToSpeech(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        return self.text_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This is the same forward method as `SeamlessM4TForTextToText`."
                "It doesn't use the text-to-unit model `SeamlessM4TTextToUnitForConditionalGeneration`."
                "If you want to generate speech, use the `.generate` method."
            )
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = attention_mask

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            spkr_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.


        Returns:
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:
            - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
              sequence_length)`and and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds"))

        if tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
        else:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
                    more languages for text translation than for speech synthesis."""
                    )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")

        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        text_generation_output = super().generate(input_ids, **kwargs_text)
        sequences = text_generation_output.sequences

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(
            self.device
        )
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids
        # second generation
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)

        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)

        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    "The speech-to-speech SeamlessM4T Model transformer which can be used for S2ST.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForSpeechToSpeech(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        return self.speech_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This is the same forward method as `SeamlessM4TForSpeechToText`. It doesn't use `self.t2u_model`."
                "If you want to generate speech, use the `generate` method."
            )

            encoder_outputs = self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                encoder_outputs[0].device
            )
            encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_features, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            spkr_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.

            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.


        Returns:
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:
            - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
              sequence_length)`and and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_features) if input_features is not None else len(kwargs.get("inputs_embeds"))

        if tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
        else:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
                    more languages for text translation than for speech synthesis."""
                    )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        text_generation_output = super().generate(input_features, **kwargs_text)
        sequences = text_generation_output.sequences

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        # get last_hidden_state from encoder
        encoder_hidden_states = self.speech_encoder(input_features=input_features, attention_mask=attention_mask)[0]

        # input modality = speech so new attention mask for the decoder
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                encoder_hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
            )

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(
            self.device
        )
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

        # second generation
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)

        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)

        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    "The original SeamlessM4T Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        current_modality (`str`, *optional*, defaults to `"text"`):
            Default modality. Used to initialize the model.
    """,
)
class SeamlessM4TModel(SeamlessM4TPreTrainedModel):
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config, current_modality="text"):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # these models already call post_init in their initialization
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def set_modality(self, modality="text"):
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    def get_encoder(self):
        if self.current_modality == "text":
            return self.text_encoder
        else:
            return self.speech_encoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_MODEL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if input_ids is None and input_features is None and inputs_embeds is None and encoder_outputs is None:
            raise ValueError(
                "`input_ids`,`input_features`, `inputs_embeds` and `encoder_outputs` are all empty. Make sure at least one of them is not."
            )
        elif input_features is not None:
            if input_ids is not None:
                logger.warning(
                    "`input_ids` is not `None` but `input_features` has been given."
                    "`input_features` will be used in priority through the `speech_encoder`. "
                    "Make sure that `input_features` and `input_ids` are mutually exclusive."
                )

            if inputs_embeds is not None:
                logger.warning(
                    "`inputs_embeds` is not `None` but `input_features` has been given."
                    "`input_features` will be used in priority through `speech_encoder`. "
                    "`inputs_embeds` will be ignored."
                )

            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This calls the same method `forward` as `SeamlessM4TForTextToText` and `SeamlessM4TForSpeechToText`"
                "depending on the input modality. If you want to generate speech, use the `generate` method."
            )

            self.set_modality("speech")

            encoder_outputs = self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif input_ids is not None or inputs_embeds is not None:
            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This calls the same method `forward` as `SeamlessM4TForTextToText` and `SeamlessM4TForSpeechToText`"
                "depending on the input modality. If you want to generate speech, use the `generate` method."
            )
            self.set_modality("text")
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = attention_mask
        # input modality = speech so new attention mask
        if self.current_modality == "speech" and attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                encoder_outputs[0].device
            )
            encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        generate_speech: Optional[bool] = True,
        **kwargs,
    ) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        """
        Generates translated token ids and/or translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively
        perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>


        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be
                ignored.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            spkr_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            generate_speech (`bool`, *optional*, defaults to `True`):
                If `False`, will only returns the text tokens and won't generate speech.

            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.

        Returns:
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor], ModelOutput]`:
            - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
            - If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of
              shape `(batch_size, sequence_length)`and and `waveform_lengths` which gives the length of each sample.
            - If `generate_speech=False`, it will returns `ModelOutput`.
        """
        if input_ids is None and input_features is None and kwargs.get("inputs_embeds", None) is None:
            raise ValueError(
                "`input_ids`,`input_features` and `inputs_embeds` are all empty. Make sure at least one of them is not."
            )

        if generate_speech and tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")

        if tgt_lang is not None:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
                    more languages for text translation than for speech synthesis."""
                    )

        batch_size = (
            len(input_features)
            if input_features is not None
            else (len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds")))
        )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            # tgt_lang gets priority over decoder input ids
            text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
            text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        if input_features is not None:
            self.set_modality("speech")
            if input_ids is not None:
                logger.warning(
                    "`input_features` and `input_ids` are both non empty. `input_features` will be used in priority "
                    "through the speech encoder. Make sure `input_features=None` if you want to use the text encoder."
                )
            text_generation_output = super().generate(input_features=input_features, **kwargs_text)
        else:
            self.set_modality("text")
            text_generation_output = super().generate(input_ids=input_ids, input_features=None, **kwargs_text)
        sequences = text_generation_output.sequences

        if not generate_speech:
            return text_generation_output

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        # get encoder last hidden states
        if self.current_modality == "speech":
            # get last_hidden_state from encoder - must do a pass through the speech encoder
            encoder_hidden_states = self.speech_encoder(
                input_features=input_features, attention_mask=attention_mask
            ).last_hidden_state

            # input modality = speech so new attention mask for the decoder
            if attention_mask is not None:
                sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                    encoder_hidden_states.device
                )
                attention_mask = _compute_new_attention_mask(
                    hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
                )
        else:
            encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(
            self.device
        )
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

        # second generation
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        # offset of control symbols
        unit_ids = torch.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)

        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)

        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


__all__ = [
    "SeamlessM4TForTextToSpeech",
    "SeamlessM4TForSpeechToSpeech",
    "SeamlessM4TForTextToText",
    "SeamlessM4TForSpeechToText",
    "SeamlessM4TModel",
    "SeamlessM4TPreTrainedModel",
    "SeamlessM4TCodeHifiGan",
    "SeamlessM4THifiGan",
    "SeamlessM4TTextToUnitForConditionalGeneration",
    "SeamlessM4TTextToUnitModel",
]
