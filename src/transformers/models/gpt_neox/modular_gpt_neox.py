from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    rotate_half,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=scaling,
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    attn_weights = attn_weights.to(value.dtype)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)

    # Reshape outputs
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GPTNeoXAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.head_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.scaling = self.head_size**-0.5
        self.is_causal = True
        self.layer_idx = layer_idx

        self._init_bias(config.max_position_embeddings)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape)
        query_states = qkv[..., : self.head_size].transpose(1, 2)
        key_states = qkv[..., self.head_size : 2 * self.head_size].transpose(1, 2)
        value_states = qkv[..., 2 * self.head_size :].transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache QKV values
        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Checking for fallbacks in case an unsupported feature is requested
        attention_type = self.config._attn_implementation
        if (output_attentions or head_mask is not None) and self.config._attn_implementation in [
            "sdpa",
            "flash_attention_2",
        ]:
            logger.warning_once(
                f"Setting `attention_type` to `eager` because `{attention_type}` does not support"
                f" `output_attentions=True` or `head_mask`."
            )
            attention_type = "eager"

        elif self.training and self.attention_dropout > 0 and self.config._attn_implementation == "flex_attention":
            logger.warning_once(
                f"Setting `attention_type` to `eager` because `dropout` is not supported in `{attention_type}`."
            )
            attention_type = "eager"

        attention_interface: Callable = eager_attention_forward
        attention_interface = (
            ALL_ATTENTION_FUNCTIONS[attention_type] if attention_type != "eager" else attention_interface
        )

        # Compute attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            head_mask=head_mask,
            **kwargs,
        )

        # Reshape outputs and final projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights


class GPTNeoXLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTNeoXAttention(config, layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        attn_output, attn_weights = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output = self.post_attention_dropout(attn_output)

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTNeoXRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class GPTNeoXPreTrainedModel(LlamaPreTrainedModel):
    base_model_prefix = "gpt_neox"
    _no_split_modules = ["GPTNeoXLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
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


GPT_NEOX_START_DOCSTRING = None  # Will be picked up by modular
GPT_NEOX_INPUTS_DOCSTRING = None  # Will be picked up by modular


class GPTNeoXModel(LlamaModel, nn.Module):
    def __init__(self, config):
        nn.Module.__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = GPTNeoXRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        converted_head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # Flex Attention converts it to a separate mask
        if head_mask is not None:
            converted_head_mask = ~converted_head_mask.bool() * torch.finfo(inputs_embeds.dtype).min
            converted_head_mask = converted_head_mask.to(dtype=self.dtype, device=self.device)
        head_mask = converted_head_mask

        hidden_states = self.emb_dropout(inputs_embeds)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        return output if return_dict else output.to_tuple()


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@add_start_docstrings(
    """GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.""", GPT_NEOX_START_DOCSTRING
)
class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.embed_out(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The GPTNeoX Model transformer with a sequence classification head on top (linear layer).

    [`GPTNeoXForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXForSequenceClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="LarsJonasson/pythia-410m-deduped-sft-swedish",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The GPT-NeoX Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
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

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "GPTNeoXForCausalLM",
    "GPTNeoXForQuestionAnswering",
    "GPTNeoXForSequenceClassification",
    "GPTNeoXForTokenClassification",
    "GPTNeoXLayer",
    "GPTNeoXModel",
    "GPTNeoXPreTrainedModel",
]
