# Follows OLMo's HF template

import math
import logging
from dataclasses import fields
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.checkpoint import checkpoint

from huggingface_hub import PyTorchModelHubMixin

from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModelForCausalLM

from .norms import get_norm_class
from .attention import get_attn_func

from .positional_embedding.head_rotary import HeadRotaryWithCast
from .positional_embedding.rotary import RotaryWithCast
from .positional_embedding.llama_rotary import LLaMARotaryWithCast
from .positional_embedding.none import identity_with_cast

from .configuration_openlm import OpenLMConfig

log = logging.getLogger(__name__)

import sys
import warnings

try:
    import xformers.ops as xops
except ModuleNotFoundError:
    warnings.warn(
        "XFormers installation not found - using native PyTorch version for operations instead.",
        RuntimeWarning
    )

def get_pos_embed(args: OpenLMConfig):
    head_dim = args.dim // args.n_heads
    if args.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, args.seq_len)
    elif args.positional_embedding_type == "llama_rotary":
        return LLaMARotaryWithCast(head_dim, args.n_heads, args.seq_len)
    elif args.positional_embedding_type == "head_rotary":
        return HeadRotaryWithCast(head_dim, args.seq_len)
    elif args.positional_embedding_type == "none":
        return identity_with_cast
    else:
        raise RuntimeError(f"Unknown positional embedding type {args.positional_embedding_type}")


class GemmaMLP(nn.Module):
    """Google's Gemma model MLP (aka GeGLU).

    Modified from https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L182-L201
    """

    def __init__(self, dim: int, hidden_dim: int, layer_id: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self._layer_id = layer_id

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.gate_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.up_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.hidden_dim)
        std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.down_proj.weight, std=std, a=-3 * std, b=3 * std)


class CustomAttn(nn.Module):
    def __init__(self, layer_id, args: OpenLMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(args.dim, 3 * args.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.pos_embed = get_pos_embed(args)
        self.attn_fn = get_attn_func(args.attn_name)
        
        self.apply_qk_norm = args.apply_qk_norm

        # initialize norm layers for queries and keys if needed
        self.q_norm = (
            get_norm_class(args.model_norm)(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            get_norm_class(args.model_norm)(
                args.n_heads * self.head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

        self.layer_id = layer_id
        self.dim = args.dim
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (self.layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor, is_causal=True, past_key_value=None, use_cache=False, attention_mask=None):
        batchsize, q_len, _ = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, q_len, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, q_len, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, q_len, self.n_heads, self.head_dim)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[1]
        queries, keys, vals = self.pos_embed(queries, keys, vals, offset=past_length)

        if past_key_value is not None and use_cache:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            vals = torch.cat([past_key_value[1], vals], dim=1)

        if use_cache:
            past_key_value = [keys, vals]

        output = self.attn_fn(
            queries,
            keys,
            vals,
            is_causal=is_causal,
            attention_mask=attention_mask,
        )

        output = output.view(batchsize, q_len, -1)

        return self.out_proj(output), past_key_value


class SwiGLUTorch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True):
        super().__init__()
        self.w12 = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        gate, x = self.w12(x).chunk(2, dim=-1)
        x = F.silu(gate) * x
        return self.w3(x)


class OpenLMBlock(nn.Module):
    def __init__(self, layer_id, args: OpenLMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads
        self.attention = CustomAttn(layer_id, args)
        self._ffn_type = args.ffn_type
        if args.ffn_type == "swiglu":
            # this follows llama / lit llama -- go to multiple of 256
            self.hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
            if "xformers.ops" in sys.modules:
                self.feed_forward = xops.SwiGLU(args.dim, self.hidden_dim, args.dim, bias=False)
            else:
                self.feed_forward = SwiGLUTorch(args.dim, self.hidden_dim, args.dim, bias=False)

        elif args.ffn_type == "swiglu_torch":
            # this follows llama / lit llama -- go to multiple of 256
            self.hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
            self.feed_forward = SwiGLUTorch(args.dim, self.hidden_dim, args.dim, bias=False)
        elif args.ffn_type == "gelu":
            # Follows mosaic mpt7b, but without a bias.
            self.hidden_dim = args.dim * 4
            self._ff_w1 = nn.Linear(args.dim, self.hidden_dim, bias=False)
            self._ff_w2 = nn.Linear(self.hidden_dim, args.dim, bias=False)
            self.feed_forward = nn.Sequential(self._ff_w1, nn.GELU(approximate="none"), self._ff_w2)
        elif args.ffn_type == "gemma_geglu":
            # this follows llama / lit llama -- go to multiple of 256
            self.hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
            self.feed_forward = GemmaMLP(args.dim, self.hidden_dim, layer_id)

        self.layer_id = layer_id
        self.attention_norm = get_norm_class(args.model_norm)(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = get_norm_class(args.model_norm)(
            args.dim,
            eps=args.norm_eps,
        )
        self.attention.seq_len = args.seq_len
        self.reset_parameters()

    def reset_parameters(self):
        if self._ffn_type == "swiglu" or self._ffn_type == "swiglu_torch":
            # initialize weights trunc_normal(1/sqrt(fan_in))
            std = 1.0 / math.sqrt(self.dim)
            torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
            # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        elif self._ffn_type == "gelu":
            std = 1.0 / math.sqrt(self.dim)
            torch.nn.init.trunc_normal_(self._ff_w1.weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self._ff_w2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, past_key_value=None, use_cache=False, attention_mask=None):
        h, past_key_value = self.attention(
            self.attention_norm(x),
            is_causal=True,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        h = x + h
        if self._ffn_type == "moe":
            ffn_out, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out, past_key_value



class OpenLMTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, params):
        super().__init__()
        # for convenience we often share param names with llama
        self.params = params
        self.dim = params.dim
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.moe_num_experts = params.moe_num_experts
        self.seq_len = params.seq_len
        self.post_embed_norm = (
            get_norm_class(params.model_norm)(
                params.dim,
                eps=params.norm_eps,
            )
            if params.post_embed_norm
            else nn.Identity()
        )
        self.weight_tying = params.weight_tying

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        ffn_type_ = params.ffn_type
        for layer_id in range(params.n_layers):
            if params.moe_freq > 0 and layer_id % params.moe_freq == 0:
                params.ffn_type = "moe"
            else:
                params.ffn_type = ffn_type_
            self.layers.append(OpenLMBlock(layer_id, params))

        # get class for normalization layers
        self.norm = get_norm_class(params.model_norm)(
            params.dim,
            eps=params.norm_eps,
        )
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        if self.weight_tying:
            self.tok_embeddings.weight = self.output.weight
        self.grad_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(self.params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False, attention_mask=None):
        """
        Args:
            input
            past_key_values
            use_cache (bool)
            attention_mask (torch.Tensor): Shape (batch_size, sequence_len), indicates tokens that should not be
                attended to. attention_mask[s, i] = False indicates that token i should not be attended to by any other
                token for sequence s.
        """
        if input_ids is not None:
            x = self.tok_embeddings(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        x = self.post_embed_norm(x)

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        elif isinstance(past_key_values, tuple):
            past_key_values = list(past_key_values)
        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing:
                x, past_key_values[i] = checkpoint(layer, x, past_key_values[i], use_cache, attention_mask)
            else:
                x, past_key_values[i] = layer(x, past_key_values[i], use_cache=use_cache, attention_mask=attention_mask)
        if past_key_values[0] is None:
            past_key_values = None
        x = self.norm(x)
        output = self.output(x)
        # follow llama in casting this to float.
        return output.float(), x, past_key_values

    def get_input_embeddings(self):
        return self.tok_embeddings

    def get_output_embeddings(self):
        return self.output


class OpenLMForCausalLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = OpenLMConfig
    base_model_prefix = "model"

    def __init__(self, config: OpenLMConfig):
        super().__init__(config)
        self.model = OpenLMTransformer(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is not None:
            raise ValueError("inputs_embeds is set but OpenLM does not support it yet")
        if attention_bias is not None:
            raise ValueError("attention_bias is et but OpenLM does not support it yet")
        if use_cache is None:
            use_cache = True
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OpenLM")
        if output_hidden_states:
            raise ValueError("output_hidden_states is not yet supported in OpenLM")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print("outer past_key_values: ", type(past_key_values))
        # if past_key_values is not None:
        #     print(len(past_key_values), type(past_key_values[0]))
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = outputs[0]
        past_key_values = outputs[2]
        hidden_states = None

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values[0][1], int):
                # This assumes that the second item of past key values is the length of the past (this is the case for linear attention)
                past_length = past_key_values[0][1]
            else:
                # This assumes that the first item of past key values is a list of all the past keys, thus the
                # shape 1 is the length of the past (this is the case for attention without window)
                past_length = past_key_values[0][0].shape[1]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.pop("use_cache", True),
        }
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.tok_embeddings

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.tok_embeddings
        else:
            return self.model.output

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.model_config.weight_tying:
            self.model.tok_embeddings = value
        else:
            self.model.output = value

    def tie_weights(self):
        """
        Copied from OLMo (description below). I removed it and the results just became garbage, so this pass is needed.
        This function is intentionally left as a no-op.
        Weight tying is handled as follows:
        - When the model is initialized, the `ff_out` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.
        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        raise NotImplementedError
