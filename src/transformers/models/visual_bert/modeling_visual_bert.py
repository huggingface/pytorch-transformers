# coding=utf-8
# Copyright 2021 The UCLA NLP Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch VisualBERT model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_visual_bert import VisualBertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VisualBertConfig"
_CHECKPOINT_FOR_DOC = "uclanlp/visualbert-vqa-coco-pre"


class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(
                    visual_embeds.size()[:-1], dtype=torch.long, device=self.position_ids.device
                )

            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = token_type_embeddings.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                    logger.warning(
                        "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                        " error."
                    )
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)

                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.size(1)} "
                            f"should be the same as `visual_embeds` length: {visual_embeds.size(1)}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, : visual_embeds.size(1), :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(
                    visual_position_ids
                )
            else:
                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)

            visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings

            embeddings = torch.cat((embeddings, visual_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in VisualBertSelfAttentionModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->VisualBert
class VisualBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = VisualBertSelfAttention(config)
        self.output = VisualBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->VisualBert
class VisualBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->VisualBert
class VisualBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VisualBertAttention(config)
        self.intermediate = VisualBertIntermediate(config)
        self.output = VisualBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisualBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VisualBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->VisualBert
class VisualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->VisualBert
class VisualBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->VisualBert
class VisualBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->VisualBert
class VisualBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VisualBertConfig
    base_model_prefix = "visual_bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
class VisualBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`VisualBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the sentence-image prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
            before SoftMax).
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
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


VISUAL_BERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VisualBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VISUAL_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
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
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        visual_embeds (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*):
            The embedded representation of the visual inputs, generally derived using using an object detector.

        visual_attention_mask (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*):
            Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        visual_token_type_ids (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*):
            Segment token indices to indicate different portions of the visual embeds.

            [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
            *visual_token_type_ids* to *1* for all tokens.

        image_text_alignment (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*):
            Image-Text alignment uses to decide the position IDs of the visual embeddings.

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
    "The bare VisualBert Model transformer outputting raw hidden-states without any specific head on top.",
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertModel(VisualBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = VisualBertEmbeddings(config)
        self.encoder = VisualBertEncoder(config)

        self.pooler = VisualBertPooler(config) if add_pooling_layer else None

        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
        from transformers import AutoTokenizer, VisualBertModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        ```"""

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

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if visual_embeds is not None:
            visual_input_shape = visual_embeds.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if visual_embeds is not None and visual_attention_mask is None:
            visual_attention_mask = torch.ones(visual_input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if visual_embeds is not None:
            combined_attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=-1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                combined_attention_mask, (batch_size, input_shape + visual_input_shape)
            )

        else:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, (batch_size, input_shape)
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
        )

        if self.bypass_transformer and visual_embeds is not None:
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_embedding_output = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :, text_length, :text_length]

            encoded_outputs = self.encoder(
                text_embedding_output,
                attention_mask=text_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoded_outputs[0]
            concatenated_input = torch.cat((sequence_output, visual_embedding_output), dim=1)
            sequence_output = self.additional_layer(concatenated_input, extended_attention_mask)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]

            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence-image prediction (classification)` head.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForPreTraining(VisualBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.visual_bert = VisualBertModel(config)
        self.cls = VisualBertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=VisualBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        sentence_image_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], VisualBertForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        sentence_image_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a matching pair of sequence A for the given image,
            - 1 indicates sequence B is a random sequence w.r.t A for the given image.

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForPreTraining

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )
        max_length = inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]
        labels = tokenizer(
            "The capital of France is Paris.", return_tensors="pt", padding="max_length", max_length=max_length
        )["input_ids"]
        sentence_image_labels = torch.tensor(1).unsqueeze(0)  # Batch_size


        outputs = model(**inputs, labels=labels, sentence_image_labels=sentence_image_labels)
        loss = outputs.loss
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            total_size = attention_mask.size(-1) + visual_attention_mask.size(-1)
            if labels.size(-1) != total_size:
                raise ValueError(
                    "The labels provided should have same sequence length as total attention mask. "
                    f"Found labels with sequence length {labels.size(-1)}, expected {total_size}."
                )

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and sentence_image_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_image_loss = loss_fct(seq_relationship_score.view(-1, 2), sentence_image_labels.view(-1))
            total_loss = masked_lm_loss + sentence_image_loss

        elif labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return VisualBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    VisualBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for VCR tasks.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForMultipleChoice(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForMultipleChoice
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        choice0 = "It is eaten with a fork and a knife."
        choice1 = "It is eaten while held in the hand."

        visual_embeds = get_visual_embeddings(image)
        # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
        visual_embeds = visual_embeds.expand(1, 2, *visual_embeds.shape)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)
        # batch size is 1
        inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
        inputs_dict.update(
            {
                "visual_embeds": visual_embeds,
                "visual_attention_mask": visual_attention_mask,
                "visual_token_type_ids": visual_token_type_ids,
                "labels": labels,
            }
        )
        outputs = model(**inputs_dict)

        loss = outputs.loss
        logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        visual_embeds = (
            visual_embeds.view(-1, visual_embeds.size(-2), visual_embeds.size(-1))
            if visual_embeds is not None
            else None
        )
        visual_attention_mask = (
            visual_attention_mask.view(-1, visual_attention_mask.size(-1))
            if visual_attention_mask is not None
            else None
        )
        visual_token_type_ids = (
            visual_token_type_ids.view(-1, visual_token_type_ids.size(-1))
            if visual_token_type_ids is not None
            else None
        )

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        _, pooled_output = outputs[0], outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled
    output) for VQA.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForQuestionAnswering
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

        text = "Who is eating the apple?"
        inputs = tokenizer(text, return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        scores = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the index of the last text token
        index_to_gather = attention_mask.sum(1) - 2  # as in original code

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # TO-CHECK: From the original code
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        )
        pooled_output = torch.gather(sequence_output, 1, index_to_gather)

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
    output) for Visual Reasoning e.g. for NLVR task.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForVisualReasoning(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)  # 2

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. A classification loss is computed (Cross-Entropy) against these labels.

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForVisualReasoning
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

        text = "Who is eating the apple?"
        inputs = tokenizer(text, return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1, Num choices 2

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        scores = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VisualBertRegionToPhraseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = 1  # config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, attention_mask):
        attention_mask = attention_mask.to(query.dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_scores = attention_scores.squeeze(1)
        return attention_scores


@add_start_docstrings(
    """
    VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment
    e.g. for Flickr30 Entities task.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = VisualBertPreTrainingHeads(config)
        self.attention = VisualBertRegionToPhraseAttention(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        region_to_phrase_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        region_to_phrase_position (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            The positions depicting the position of the image embedding corresponding to the textual tokens.

        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length, visual_sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. KLDivLoss is computed against these labels and the
            outputs from the attention layer.

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForRegionToPhraseAlignment
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForRegionToPhraseAlignment.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        text = "Who is eating the apple?"
        inputs = tokenizer(text, return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        region_to_phrase_position = torch.ones((1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]))

        inputs.update(
            {
                "region_to_phrase_position": region_to_phrase_position,
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        labels = torch.ones(
            (1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2], visual_embeds.shape[-2])
        )  # Batch size 1

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        scores = outputs.logits
        ```"""
        if region_to_phrase_position is None:
            raise ValueError("`region_to_phrase_position` should not be None when using Flickr Model.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        region_to_phrase_position_mask = (region_to_phrase_position != -1).long()

        # Make the -1 become 0
        region_to_phrase_position = region_to_phrase_position * region_to_phrase_position_mask

        # Selected_positions = batch x selected position x dim
        expanded_region_to_phrase_positions = region_to_phrase_position.unsqueeze(2).expand(
            region_to_phrase_position.size(0), region_to_phrase_position.size(1), sequence_output.size(2)
        )
        selected_positions = sequence_output.gather(1, expanded_region_to_phrase_positions)

        # Visual Features = batch x visual_feature_length x dim
        # This will need separate image and visual masks.
        visual_features = sequence_output[:, attention_mask.size(1) :]

        if visual_features.size(1) != visual_attention_mask.size(1):
            raise ValueError(
                f"Visual features length :{visual_features.size(1)} should be the same"
                f" as visual attention mask length: {visual_attention_mask.size(1)}."
            )

        logits = self.attention(selected_positions, visual_features, visual_attention_mask)

        loss = None

        if labels is not None:
            # scores = batch x selected position x visual_feature
            # scores = selected_positions.bmm(visual_features.transpose(1,2))
            # label = batch x selected_postion x needed position
            loss_fct = KLDivLoss(reduction="batchmean")
            log_softmax = LogSoftmax(dim=-1)
            scores = log_softmax(logits)
            labels = labels.contiguous()
            loss = loss_fct(scores, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "VisualBertForMultipleChoice",
    "VisualBertForPreTraining",
    "VisualBertForQuestionAnswering",
    "VisualBertForRegionToPhraseAlignment",
    "VisualBertForVisualReasoning",
    "VisualBertLayer",
    "VisualBertModel",
    "VisualBertPreTrainedModel",
]
