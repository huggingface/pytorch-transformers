# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
"""PyTorch Flaubert model, based on XLM."""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import gelu
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary, SQuADHead
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_flaubert import FlaubertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "flaubert/flaubert_base_cased"
_CONFIG_FOR_DOC = "FlaubertConfig"


# Copied from transformers.models.xlm.modeling_xlm.create_sinusoidal_embeddings
def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()


# Copied from transformers.models.xlm.modeling_xlm.get_masks
def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = lengths.size(0)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


# Copied from transformers.models.xlm.modeling_xlm.MultiHeadAttention
class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)  # (bs, n_heads, qlen, klen)

        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


# Copied from transformers.models.xlm.modeling_xlm.TransformerFFN
class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if config.gelu_activation else nn.functional.relu
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


FLAUBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FLAUBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Length of each sentence that can be used to avoid performing attention on padding token indices. You can
            also use `attention_mask` for the same result (see above), kept here for compatibility. Indices selected in
            `[0, ..., input_ids.size(-1)]`:
        cache (`Dict[str, torch.FloatTensor]`, *optional*):
            Dictionary strings to `torch.FloatTensor` that contains precomputed hidden-states (key and values in the
            attention blocks) as computed by the model (see `cache` output below). Can be used to speed up sequential
            decoding. The dictionary object will be modified in-place during the forward pass to add newly computed
            hidden-states.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
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
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
# Copied from transformers.models.xlm.modeling_xlm.XLMPredLayer with XLM->Flaubert
class FlaubertPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super().__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim

        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs
            if y is not None:
                loss = nn.functional.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction="mean")
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        return outputs


# Copied from transformers.models.xlm.modeling_xlm.XLMPreTrainedModel with XLM->Flaubert
class FlaubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlaubertConfig
    load_tf_weights = None
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @property
    def dummy_inputs(self):
        inputs_list = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, FlaubertModel) and self.config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(
                self.config.max_position_embeddings, self.config.emb_dim, out=module.position_embeddings.weight
            )


class FlaubertModel(FlaubertPreTrainedModel):
    def __init__(self, config):  # , dico, is_encoder, with_output):
        super().__init__(config)

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError("Currently Flaubert can only be used as an encoder")
        # self.with_output = with_output
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        # self.dico = dico
        # self.id2lang = config.id2lang
        # self.lang2id = config.lang2id
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = config.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads  # 8 by default
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=config.layer_norm_eps)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        # if self.is_decoder:
        #     self.layer_norm15 = nn.ModuleList()
        #     self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, config=config))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            # if self.is_decoder:
            #     self.layer_norm15.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            #     self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, config=config))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))

        if hasattr(config, "pruned_heads"):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for layer, heads in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})

        # Initialize weights and apply final processing
        self.post_init()

        self.layerdrop = getattr(config, "layerdrop", 0.0)
        self.pre_norm = getattr(config, "pre_norm", False)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # Copied from transformers.models.xlm.modeling_xlm.XLMModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings

    # Copied from transformers.models.xlm.modeling_xlm.XLMModel.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    # Copied from transformers.models.xlm.modeling_xlm.XLMModel._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        lengths: Optional[torch.LongTensor] = None,
        cache: Optional[Dict[str, torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # removed: src_enc=None, src_len=None
        if input_ids is not None:
            bs, slen = input_ids.size()
        else:
            bs, slen = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = torch.tensor([slen] * bs, device=device)
        # mask = input_ids != self.pad_index

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if position_ids is None:
            if hasattr(self, "position_ids"):
                position_ids = self.position_ids[:, :slen]
                position_ids = position_ids.expand((bs, slen))
            else:
                position_ids = torch.arange(slen, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand((bs, slen))
        else:
            assert position_ids.size() == (bs, slen)  # (slen, bs)
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (bs, slen)  # (slen, bs)
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb and self.config.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            # LayerDrop
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            if not self.pre_norm:
                attn_outputs = self.attentions[i](
                    tensor,
                    attn_mask,
                    cache=cache,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
                attn = attn_outputs[0]
                if output_attentions:
                    attentions = attentions + (attn_outputs[1],)
                attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm1[i](tensor)
            else:
                tensor_normalized = self.layer_norm1[i](tensor)
                attn_outputs = self.attentions[i](tensor_normalized, attn_mask, cache=cache, head_mask=head_mask[i])
                attn = attn_outputs[0]
                if output_attentions:
                    attentions = attentions + (attn_outputs[1],)
                attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            if not self.pre_norm:
                tensor = tensor + self.ffns[i](tensor)
                tensor = self.layer_norm2[i](tensor)
            else:
                tensor_normalized = self.layer_norm2[i](tensor)
                tensor = tensor + self.ffns[i](tensor_normalized)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)


@add_start_docstrings(
    """
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    FLAUBERT_START_DOCSTRING,
)
# Copied transformers.models.xlm.modeling_xlm.XLMWithLMHeadModel with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertWithLMHeadModel(FlaubertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["pred_layer.proj.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.pred_layer = FlaubertPredLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.pred_layer.proj

    def set_output_embeddings(self, new_embeddings):
        self.pred_layer.proj = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Overwritten -- uses a language id

        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = input_ids.shape[0]
        mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, mask_token], dim=1)
        if lang_id is not None:
            langs = torch.full_like(input_ids, lang_id)
        else:
            langs = None
        return {"input_ids": input_ids, "langs": langs}

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<special1>",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# Copied transformers.models.xlm.modeling_xlm.XLMForSequenceClassification with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForSequenceClassification(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = FlaubertModel(config)
        self.sequence_summary = SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# Copied from transformers.models.xlm.modeling_xlm.XLMForTokenClassification with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForTokenClassification(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = FlaubertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAUBERT_START_DOCSTRING,
)
# Copied from transformers.models.xlm.modeling_xlm.XLMForQuestionAnsweringSimple with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForQuestionAnsweringSimple(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = FlaubertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    Flaubert Model with a beam-search span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAUBERT_START_DOCSTRING,
)
@dataclass
# Copied from transformer.models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput with XLM->Flaubert
class FlaubertForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a `SquadHead`.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformer.models.xlm.modeling_xlm.XLMForQuestionAnswering with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForQuestionAnswering(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = FlaubertModel(config)
        self.qa_outputs = SQuADHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FlaubertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        is_impossible: Optional[torch.Tensor] = None,
        cls_index: Optional[torch.Tensor] = None,
        p_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaubertForQuestionAnsweringOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the classification token to use as input for computing plausibility of the
            answer.
        p_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
            masked. 0.0 mean token is not masked.

        Returns:

        Example:

        ```python
        >>> from transformers import XLMTokenizer, XLMForQuestionAnswering
        >>> import torch

        >>> tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
        >>> model = XLMForQuestionAnswering.from_pretrained("FacebookAI/xlm-mlm-en-2048")

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
        ...     0
        ... )  # Batch size 1
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]

        outputs = self.qa_outputs(
            output,
            start_positions=start_positions,
            end_positions=end_positions,
            cls_index=cls_index,
            is_impossible=is_impossible,
            p_mask=p_mask,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return FlaubertForQuestionAnsweringOutput(
            loss=outputs.loss,
            start_top_log_probs=outputs.start_top_log_probs,
            start_top_index=outputs.start_top_index,
            end_top_log_probs=outputs.end_top_log_probs,
            end_top_index=outputs.end_top_index,
            cls_logits=outputs.cls_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# Copied from transformer.models.xlm.modeling_xlm.XLMForMultipleChoice with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForMultipleChoice(FlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.transformer = FlaubertModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        langs = langs.view(-1, langs.size(-1)) if langs is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the Flaubert multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "FlaubertForMultipleChoice",
    "FlaubertForQuestionAnswering",
    "FlaubertForQuestionAnsweringSimple",
    "FlaubertForSequenceClassification",
    "FlaubertForTokenClassification",
    "FlaubertModel",
    "FlaubertWithLMHeadModel",
    "FlaubertPreTrainedModel",
]
