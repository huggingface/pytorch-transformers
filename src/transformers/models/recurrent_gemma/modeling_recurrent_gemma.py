# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
"""PyTorch RecurrentGemma model."""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torchdynamo_compiling
from .configuration_recurrent_gemma import RecurrentGemmaConfig


logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "RecurrentGemmaConfig"
_MAX_SQRT_GRADIENT = 1000.0


# Copied from transformers.models.gemma.modeling_gemma.GemmaRMSNorm with Gemma->RecurrentGemma
class RecurrentGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst RecurrentGemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


ALL_LAYERNORM_LAYERS.append(RecurrentGemmaRMSNorm)


class RecurrentGemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RecurrentGemmaSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.partial_rotary_factor = config.partial_rotary_factor

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=True)
        self.rotary_emb = RecurrentGemmaRotaryEmbedding(
            int(self.partial_rotary_factor * self.head_dim),
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)

        # Partial rotary embedding
        query_rot, query_pass = torch.chunk(query_states, int(1 / self.partial_rotary_factor), dim=-1)
        key_rot, key_pass = torch.chunk(key_states, int(1 / self.partial_rotary_factor), dim=-1)
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if use_cache and hasattr(self, "key_states"):
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = self._update_cache(key_states, value_states, **cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=causal_mask,  # pretty much a must for sliding window backend!
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.head_dim**-0.5,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _setup_cache(self, batch_size, device, dtype=None):
        if dtype is None and self.config.torch_dtype is not None:
            dtype = self.config.torch_dtype
        dtype = dtype if dtype is not None else torch.float32
        cache_shape = (batch_size, self.num_key_value_heads, self.config.attention_window_size, self.head_dim)
        self.value_states = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.key_states = torch.zeros(cache_shape, dtype=dtype, device=device)

    @torch.no_grad()
    def _update_cache(self, key_states, value_states, **cache_kwargs):
        """
        torch.compile compatible sliding window.
        Computes the `indices` based on `cache_position >= self.config.attention_window_size - 1`.
        The `to_shift` is only true once we are above attention_window_size. Thus with `attention_window_size==64`:

        indices = (slicing + to_shift[-1].int()-1) % self.config.attention_window_size
        tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

        We overwrite the cache using these, then we always write at cache_position (clamped to `attention_window_size`)
        """
        cache_position = cache_kwargs.get("cache_position")
        if cache_position.shape[0] > self.config.attention_window_size:
            # int indexing -> device sync? in compile, use tensor
            k_out = key_states[:, :, -self.config.attention_window_size :, :]
            v_out = value_states[:, :, -self.config.attention_window_size :, :]
        else:
            slicing = torch.ones(
                self.config.attention_window_size, dtype=torch.long, device=value_states.device
            ).cumsum(0)
            cache_position = cache_position.clamp(0, self.config.attention_window_size - 1)
            to_shift = cache_position >= self.config.attention_window_size - 1
            indices = (slicing + to_shift[-1].int() - 1) % self.config.attention_window_size

            k_out, v_out = self.key_states.to(key_states.device), self.value_states.to(value_states.device)
            k_out = k_out[:, :, indices]
            v_out = v_out[:, :, indices]

            k_out[:, :, cache_position] = key_states.to(k_out.dtype)
            v_out[:, :, cache_position] = value_states.to(v_out.dtype)

        self.key_states, self.value_states = k_out, v_out
        return k_out, v_out


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class RecurrentGemmaRglru(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.block_width = config.lru_width // self.num_attention_heads

        self.recurrent_param = nn.Parameter(torch.empty([config.lru_width]))
        self.input_gate_weight = nn.Parameter(
            torch.empty([self.num_attention_heads, self.block_width, self.block_width])
        )
        self.input_gate_bias = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width]))

        self.recurrent_gate_weight = nn.Parameter(
            torch.empty([self.num_attention_heads, self.block_width, self.block_width])
        )
        self.recurrent_gate_bias = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width]))
        self.recurrent_states = None

    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, lru_width = activations.shape
        reset = position_ids[:, :, None] == 0

        reshape_act = activations.reshape(batch_size * seq_len, self.num_attention_heads, self.block_width)
        reshape_act = reshape_act.permute(1, 0, 2)

        res = torch.baddbmm(self.input_gate_bias[:, None, :], reshape_act, self.input_gate_weight)
        input_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

        res = torch.baddbmm(self.recurrent_gate_bias[:, None, :], reshape_act, self.recurrent_gate_weight)
        recurrent_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

        # Compute the parameter `A` of the recurrence.
        log_recurrent_gate = -8.0 * recurrent_gate * nn.functional.softplus(self.recurrent_param)
        recurrent_gate = torch.exp(log_recurrent_gate)
        a_square = torch.exp(2 * log_recurrent_gate)

        # Gate the input.
        gated_inputs = activations * input_gate

        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16. TODO a bit annoying
        multiplier = 1
        tracing = isinstance(activations, torch.fx.Proxy) or is_torchdynamo_compiling()
        if not torch.jit.is_tracing() and not tracing:
            multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset + ~reset * multiplier
        normalized_x = gated_inputs * multiplier.type(activations.dtype)

        hidden_states, recurrent_states = self._rnn_scan(
            hidden_states=normalized_x,
            recurrent_gate=recurrent_gate,
            reset=reset,
            recurrent_states=self.recurrent_states,
        )
        self.recurrent_states = recurrent_states
        return hidden_states

    # TODO refactor
    def _rnn_scan(
        self,
        hidden_states: torch.Tensor,
        recurrent_gate: torch.Tensor,
        reset: torch.Tensor,
        recurrent_states: Union[torch.Tensor, None],
        acc_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the recurrence of a linear RNN.

        Args:
        hidden_states: The input sequence.
        recurrent_gate: The diagonal of the recurrence matrix `A`.
        reset: Indicator of document boundaries, e.g. when to reset the hidden state
            of the RNN.
        recurrent_states: The initial hidden state.
        acc_dtype: The data type for the accumulation.

        Returns:
        The output of the linear recurrence.
        """
        # Multiply `a` by the reset.
        recurrent_gate = recurrent_gate * ~reset

        if hidden_states.shape[1] == 1:
            # Using scan in sampling mode.
            if recurrent_states is None:  # same here, when decoding you always have cache
                return hidden_states, hidden_states[:, 0].type(acc_dtype)

            else:
                contextualized_states = recurrent_gate.type(acc_dtype) * recurrent_states[:, None].to(
                    recurrent_gate.device
                )
                contextualized_states += hidden_states.type(acc_dtype)
                return contextualized_states.type(hidden_states.dtype), contextualized_states[:, -1]

        else:
            # Using scan in linear mode.
            if recurrent_states is None:
                recurrent_states = torch.zeros(hidden_states[:, 0].shape, dtype=acc_dtype, device=hidden_states.device)

            contextualized_states = torch.zeros_like(hidden_states)
            for t in range(hidden_states.shape[1]):
                recurrent_states = recurrent_gate[:, t].type(acc_dtype) * recurrent_states.to(recurrent_gate.device)
                recurrent_states = recurrent_states + hidden_states[:, t].type(acc_dtype)
                contextualized_states[:, t] = recurrent_states.type(hidden_states.dtype)

        return contextualized_states, recurrent_states


class RecurrentGemmaRecurrentBlock(nn.Module):
    """Griffin and Hawk's recurrent block."""

    def __init__(self, config):
        super().__init__()
        self.lru_width = config.lru_width
        self.hidden_size = config.hidden_size
        self.linear_y = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_x = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_out = nn.Linear(in_features=config.lru_width, out_features=config.hidden_size)
        self.conv1d_width = config.conv1d_width
        self.conv_1d = nn.Conv1d(
            config.lru_width,
            config.lru_width,
            kernel_size=config.conv1d_width,
            groups=config.lru_width,
            padding=config.conv1d_width - 1,
        )
        self.rg_lru = RecurrentGemmaRglru(config)
        self.act_fn = ACT2FN[config.hidden_activation]

        self.conv1d_state = None

    def forward(
        self,
        input_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, seq_len, _ = input_states.shape

        y_branch = self.linear_y(input_states)
        y_branch = self.act_fn(y_branch)

        x_branch = self.linear_x(input_states)
        x_branch = x_branch.transpose(1, 2)

        if use_cache:
            if cache_position.shape[0] != 1:  # prefill
                self.conv1d_state = nn.functional.pad(x_branch, (self.conv1d_width - x_branch.shape[-1] - 1, 0))
                x_branch = self.conv_1d(x_branch)[..., :seq_len]
            else:  # decoding
                conv_state = torch.cat((self.conv1d_state, x_branch), -1)
                x_branch = torch.sum(conv_state * self.conv_1d.weight[:, 0, :], dim=-1) + self.conv_1d.bias
                x_branch = x_branch.unsqueeze(-1)
                self.conv1d_state = conv_state[:, :, 1:]
        else:
            x_branch = self.conv_1d(x_branch)[..., :seq_len]

        x_branch = self.rg_lru(x_branch.transpose(1, 2), position_ids)

        hidden_states = x_branch * y_branch
        hidden_states = self.linear_out(hidden_states)
        return hidden_states

    def _setup_cache(self, batch, device, dtype):
        # recurrent_states always computed in full precision
        self.rg_lru.recurrent_states = torch.zeros((batch, self.lru_width), device=device, dtype=torch.float32)
        self.conv1d_state = torch.zeros((batch, self.hidden_size, self.conv1d_width - 1), device=device, dtype=dtype)


TEMPORAL_BLOCK_CLASSES = {"recurrent": RecurrentGemmaRecurrentBlock, "attention": RecurrentGemmaSdpaAttention}


class RecurrentGemmaMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // 2
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states):
        gate = self.act_fn(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


class RecurrentGemmaDecoderLayer(nn.Module):
    """Griffin and Hawk's residual block."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.temporal_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.temporal_block = TEMPORAL_BLOCK_CLASSES[config.layers_block_type[layer_idx]](config)
        self.channel_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_block = RecurrentGemmaMlp(config)

    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor = None,
        use_cache: bool = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_activations = activations
        inputs_normalized = self.temporal_pre_norm(raw_activations)  # RMSNorm introduces slight slight differences

        hidden_states = self.temporal_block(
            inputs_normalized, position_ids, attention_mask, cache_position=cache_position, use_cache=use_cache
        )

        residual = hidden_states + raw_activations

        hidden_states = self.channel_pre_norm(residual)
        hidden_states = self.mlp_block(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states


RECURRENTGEMMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RecurrentGemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.",
    RECURRENTGEMMA_START_DOCSTRING,
)
class RecurrentGemmaPreTrainedModel(PreTrainedModel):
    config_class = RecurrentGemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RecurrentGemmaDecoderLayer"]
    _skip_keys_device_placement = ["cache"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False  # we can't compare with eager for now
    _supports_cache_class = True
    _supports_quantized_cache = True

    def _init_weights(self, module):
        std = math.sqrt(self.config.w_init_variance_scale / self.config.conv1d_width)
        if isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RecurrentGemmaSdpaAttention):
            torch.nn.init.normal_(module.q_proj.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            torch.nn.init.normal_(module.k_proj.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            torch.nn.init.normal_(module.v_proj.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))

            std = math.sqrt(self.config.final_w_init_variance_scale / self.config.hidden_size)
            torch.nn.init.normal_(module.o_proj.weight, mean=0.0, std=std)
        elif isinstance(module, RecurrentGemmaRecurrentBlock):
            torch.nn.init.zeros_(module.linear_x.bias)
            torch.nn.init.normal_(module.linear_x.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))

            torch.nn.init.zeros_(module.linear_y.bias)
            torch.nn.init.normal_(module.linear_y.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))

            std = math.sqrt(self.config.final_w_init_variance_scale / self.config.lru_width)
            torch.nn.init.normal_(module.linear_out.weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.linear_out.bias)
        elif isinstance(module, RecurrentGemmaRglru):
            std = math.sqrt(
                self.config.w_init_variance_scale / (self.config.lru_width // self.config.num_attention_heads)
            )
            torch.nn.init.normal_(module.input_gate_weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.recurrent_gate_weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.input_gate_bias)
            torch.nn.init.zeros_(module.recurrent_gate_bias)

            module.recurrent_param.data.uniform_(0.9**2 + 1e-8, 0.999**2 + 1e-8)
            module.recurrent_param.data.log_().mul_(0.5)
            module.recurrent_param.data.neg_().exp_().sub_(1.0).log_()
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)

    def _setup_cache(self, config, batch, device, dtype):
        layers = getattr(self, "model", self).layers
        for layer in layers:
            layer.temporal_block._setup_cache(batch, device, dtype)

    def reset_cache(self, batch, device, dtype):
        pass


RECURRENTGEMMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all  See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.",
    RECURRENTGEMMA_START_DOCSTRING,
)
class RecurrentGemmaModel(RecurrentGemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`RecurrentGemmaDecoderLayer`]

    Args:
        config: RecurrentGemmaConfig
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RecurrentGemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.register_buffer(
            "normalizer", torch.tensor(self.config.hidden_size**0.5, dtype=torch.bfloat16), persistent=False
        )
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and inputs_embeds.shape[1] != 1:  # TODO let's maybe only call in the `generate`?
            self._setup_cache(self.config, hidden_states.shape[0], hidden_states.device, hidden_states.dtype)

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        hidden_states = hidden_states * self.normalizer.type(hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        for i, residual_block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    residual_block.__call__, hidden_states, position_ids, causal_mask, cache_position, use_cache
                )
            else:
                hidden_states = residual_block(hidden_states, position_ids, causal_mask, cache_position, use_cache)

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    # Ignore copy
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = max(self.config.attention_window_size, sequence_length)

        diagonal = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = diagonal
        if sequence_length != 1:
            causal_mask = torch.triu(diagonal, diagonal=-1)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if attention_mask is not None and attention_mask.device.type == "cuda":
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# TODO: re-enable check: Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->RECURRENTGEMMA,Llama->RecurrentGemma,llama->gemma
class RecurrentGemmaForCausalLM(RecurrentGemmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RecurrentGemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RecurrentGemmaForCausalLM

        >>> model = RecurrentGemmaForCausalLM.from_pretrained("google/recurrentgemma-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Soft-cap the logits TODO remove if always done.
        # if self.config.logits_soft_cap is not None:
        cap = self.config.logits_soft_cap
        logits = nn.functional.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    # Ignore copy
    def _reorder_cache(self, past_key_values, beam_idx):
        for layer in self.layers:
            if hasattr(layer.temporal_block, "key_states"):
                k_state = layer.temporal_block.key_states
                v_state = layer.temporal_block.value_states
                k_state = k_state.index_select(0, beam_idx.to(k_state.device))
                v_state = v_state.index_select(0, beam_idx.to(v_state.device))
        return None
