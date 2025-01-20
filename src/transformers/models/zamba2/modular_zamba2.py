# coding=utf-8
# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
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
import math
from itertools import cycle
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from ...utils.import_utils import (
    is_causal_conv1d_available,
    is_mamba_ssm_available,
)
from ..llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from ..mamba2.modeling_mamba2 import MambaRMSNormGated, pad_tensor_by_size, reshape_into_chunks, segment_sum
from ..zamba.modeling_zamba import (
    ZambaAttention,
    ZambaAttentionDecoderLayer,
    ZambaForCausalLM,
    ZambaForSequenceClassification,
    ZambaHybridDynamicCache,
    ZambaHybridLayer,
    ZambaMambaDecoderLayer,
    ZambaMLP,
    ZambaModel,
    ZambaRMSNorm,
    eager_attention_forward,
)
from .configuration_zamba2 import Zamba2Config


if is_mamba_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


_CONFIG_FOR_DOC = "Zyphra/Zamba2-2.7B"

logger = logging.get_logger(__name__)


class Zamba2RMSNormGated(MambaRMSNormGated):
    pass


class Zamba2RMSNorm(ZambaRMSNorm):
    pass


def count_mem_blocks_in_config(config: Zamba2Config):
    """
    Count number of shared blocks
    """
    num_gs = 0
    for val in config.layers_block_type:
        if val == "hybrid":
            num_gs += 1
    return num_gs


def layer_type_list(config: Zamba2Config):
    """
    Returns list of layer ids containing hybrid layers
    """
    output_list = []
    for index, type in enumerate(config.layers_block_type):
        if type == "hybrid":
            output_list.append(index)
    return output_list


class Zamba2HybridDynamicCache(ZambaHybridDynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(
        self, config: Zamba2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.n_mamba_heads = config.n_mamba_heads
        self.transformer_layers = []
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.mamba_ngroups * config.mamba_d_state,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(
                batch_size, self.n_mamba_heads, config.mamba_headdim, self.ssm_state_size, device=device, dtype=dtype
            )
            for i in range(config.num_hidden_layers)
        }
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "hybrid":
                self.transformer_layers.append(i)
        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
            return 0
        return self.key_cache[layer_idx].shape[-2]


class Zamba2RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        config: Zamba2Config,
        device=None,
    ):
        super().__init__(config, device)
        # we cannot use the config here to parameterize because of a factor 2 for the head_dim
        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=config.rope_theta, dim=config.attention_head_dim
        )


class Zamba2Attention(ZambaAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
    The input dimension here is attention_hidden_size = 2 * hidden_size, and head_dim = attention_hidden_size // num_heads.
    The extra factor of 2 comes from the input being the concatenation of original_hidden_states with the output of the previous (mamba) layer
    (see fig. 2 in https://arxiv.org/pdf/2405.16712).
    Additionally, replaced
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) with
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim/2)
    Finally, this attention layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this
    layer is tied, un-tied adapters (formally the same as LoRA but used in the base model) modules are added to the q, k, v projectors to increase
    expressivity with a small memory overhead (see Fig. 2 of https://arxiv.org/pdf/2411.15242).
    """

    def __init__(
        self,
        config: Zamba2Config,
        layer_idx: Optional[int] = None,
        num_fwd_mem_blocks: int = None,
        block_id: int = None,
    ):
        super().__init__(config, layer_idx)
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.layer_block_map = layer_type_list(config)
        self.block_id = block_id

        if config.use_shared_attention_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])

            for i in range(self.num_fwd_mem_blocks):
                if i % config.num_mem_blocks == block_id:
                    linear_q_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                    linear_k_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                    linear_v_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                else:
                    linear_q_adapter = nn.Identity()
                    linear_k_adapter = nn.Identity()
                    linear_v_adapter = nn.Identity()
                self.linear_q_adapter_list.append(linear_q_adapter)
                self.linear_k_adapter_list.append(linear_k_adapter)
                self.linear_v_adapter_list.append(linear_v_adapter)

        self.layer_dic = {value: index for index, value in enumerate(self.layer_block_map)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Zamba2HybridDynamicCache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.config.use_shared_attention_adapter:
            adapter_layer_idx = self.layer_dic[layer_idx]
            query_states = query_states + self.linear_q_adapter_list[adapter_layer_idx](hidden_states)
            key_states = key_states + self.linear_k_adapter_list[adapter_layer_idx](hidden_states)
            value_states = value_states + self.linear_v_adapter_list[adapter_layer_idx](hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Zamba2MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Zamba2Config, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * self.hidden_size)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = "silu"
        self.act = nn.SiLU()

        self.n_groups = config.mamba_ngroups
        self.head_dim = config.mamba_headdim
        self.num_heads = self.config.n_mamba_heads
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=config.mamba_d_conv,
            groups=self.conv_dim,
            padding=config.mamba_d_conv - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.add_bias_linear,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = Zamba2RMSNormGated(self.intermediate_size, eps=1e-5)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Zamba2HybridDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # getting projected states from cache if it exists
        if cache_params is not None and cache_params.has_previous_state:
            in_projected_states = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
            _, _, gate, hidden_states_B_C, dt = torch.split(in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())  # (nheads,)

            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        # if no cache is found, calling the kernel
        else:
            if attention_mask is not None and not torch.all(attention_mask == 1):
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            # 1. Gated MLP's linear projection
            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
            if attention_mask is not None:
                input_not_masked = torch.all(attention_mask == 1)
            else:
                input_not_masked = True

            if self.training and cache_params is None and input_not_masked:
                out, ssm_state = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                # 1D Convolution
                if cache_params is not None:
                    hidden_states_B_C_t = hidden_states_B_C.transpose(1, 2)
                    conv_state = nn.functional.pad(
                        hidden_states_B_C_t, (self.conv_kernel_size - hidden_states_B_C_t.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_state)
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
                    )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)[:, :seq_len]
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    time_step,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[Zamba2HybridDynamicCache]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # Gated MLP's linear projection
        if cache_params is not None and cache_params.has_previous_state:
            projected_states =  self.in_proj(input_states.squeeze(1))
        else:
            if attention_mask is not None and not torch.all(attention_mask==1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    input_states = (input_states * attention_mask[:, :, None]).to(dtype)
            projected_states =  self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size -  2 * self.n_groups * self.ssm_state_size- self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )

        # Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            if cache_params.has_previous_state:
                gate = gate.unsqueeze(1)
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                # handle batched generation - states are copied through
                conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]         # [batch, 1, intermediate_size] : decoding
            else:
                hidden_states = hidden_states.transpose(1,2)
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states).transpose(1,2))[:, :seq_len, :]     # [batch, intermediate_size, seq_len]
                if attention_mask is not None and not torch.all(attention_mask==1):
                    dtype = hidden_states.dtype
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if cache_params is not None and cache_params.has_previous_state:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = torch.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len,  -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]


            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)


            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]  * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
            if cache_params is not None and cache_params.has_previous_state:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            # compute Yoff
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Zamba2HybridDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

        return self.torch_forward(hidden_states, cache_params, attention_mask)


class Zamba2MLP(ZambaMLP):
    def __init__(self, config: Zamba2Config, num_fwd_mem_blocks=None, block_id: int = None):
        """
        This MLP layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this layer
        is tied, un-tied adapter modules (formally same as LoRA, but used in the base model) are added to the up and gate projectors to increase expressivity with a small memory overhead.
        """
        super().__init__(config)
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.block_id = block_id

        del self.gate_proj
        del self.up_proj
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.add_bias_linear)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)

        if self.config.use_shared_mlp_adapter:
            self.gate_up_proj_adapter_list = nn.ModuleList([])
            for i in range(self.num_fwd_mem_blocks):
                if i % config.num_mem_blocks == block_id:
                    gate_up_proj_adapter = nn.Sequential(
                        nn.Linear(self.config.hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, 2 * self.intermediate_size, bias=False),
                    )
                else:
                    gate_up_proj_adapter = nn.Identity()
                self.gate_up_proj_adapter_list.append(gate_up_proj_adapter)

        layer_block_map = layer_type_list(config)
        self.layer_dic = {value: index for index, value in enumerate(layer_block_map)}

    def forward(self, hidden_state, layer_idx=None):
        gate_up_state = self.gate_up_proj(hidden_state)
        if self.config.use_shared_mlp_adapter:
            layer_idx = self.layer_dic[layer_idx]
            gate_up_state = gate_up_state + self.gate_up_proj_adapter_list[layer_idx](hidden_state)

        gate_up_state = torch.chunk(gate_up_state, 2, dim=-1)
        hidden_state = self.act_fn(gate_up_state[0]) * gate_up_state[1]
        output = self.down_proj(hidden_state)
        return output


class Zamba2AttentionDecoderLayer(ZambaAttentionDecoderLayer):
    def __init__(self, config: Zamba2Config, block_id: int = None, layer_idx: Optional[int] = None):
        self.block_id = block_id
        num_gs = count_mem_blocks_in_config(config)
        super().__init__(config, layer_idx)
        self.self_attn = Zamba2Attention(config, layer_idx=-1, num_fwd_mem_blocks=num_gs, block_id=block_id)
        self.feed_forward = Zamba2MLP(config, num_fwd_mem_blocks=num_gs, block_id=block_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Zamba2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): output of previous Mamba layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output of shape `(batch, seq_len, embed_dim)`.
                This is concatenated with `hidden_states` (which is the output of the previous (mamba) layer). The
                concatenated tensor is then used as input of the pre-attention RMSNorm
                (see fig. 2 in https://arxiv.org/pdf/2405.16712).
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Zamba2HybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, layer_idx)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Zamba2MambaDecoderLayer(ZambaMambaDecoderLayer):
    def __init__(self, config: Zamba2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mamba = Zamba2MambaMixer(config=config, layer_idx=layer_idx)
        self.input_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Zamba2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
        transformer_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Zamba2HybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            transformer_hidden_states (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Output of the previous shared transformer layer (if present) of shape `(batch_size, seq_len, embed_dim)`.
        """

        residual = hidden_states

        # `transformer_hidden_states` is the output from shared transformer + linear layer (see fig. 2 in https://arxiv.org/pdf/2405.16712).
        # `transformer_hidden_states` is then added to the input to the mamba layer below (as described in eq. (6) of https://arxiv.org/pdf/2405.16712).
        hidden_states = (
            hidden_states + transformer_hidden_states if transformer_hidden_states is not None else hidden_states
        )
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_value,
            attention_mask=attention_mask,
        )

        self_attn_weights = None

        # residual connection after mamba
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class Zamba2HybridLayer(ZambaHybridLayer):
    def __init__(
        self, shared_transformer: Zamba2AttentionDecoderLayer, linear: nn.Linear, mamba: Zamba2MambaDecoderLayer
    ):
        super().__init__(shared_transformer, linear, mamba)
        del self.shared_transf
        self.shared_transformer = shared_transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Zamba2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output that will be concatenated with
            hidden activations to form the input of the shared transformer layer.
            layer_idx (`int`): layer number.
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Zamba2HybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """

        layer_outputs = self.shared_transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )

        transformer_hidden_states = layer_outputs[0]

        if output_attentions:
            self_attn_weights = layer_outputs[1]

        transformer_hidden_states = self.linear(transformer_hidden_states)

        layer_outputs = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        if output_attentions:
            layer_outputs = (layer_outputs[0], self_attn_weights) + layer_outputs[2:]

        return layer_outputs


ZAMBA2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Zamba2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Zamba2 Model outputting raw hidden-states without any specific head on top.",
    ZAMBA2_START_DOCSTRING,
)
class Zamba2PreTrainedModel(PreTrainedModel):
    config_class = Zamba2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Zamba2AttentionDecoderLayer", "Zamba2MambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = False
    _supports_cache_class = True  # Note: only supports Zamba2HybridDynamicCache
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Zamba2MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.n_mamba_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True


ZAMBA2_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Zamba2HybridDynamicCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A Zamba2HybridDynamicCache object containing pre-computed hidden-states (keys and values in the
            self-attention blocks and convolution and ssm states in the mamba blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
            Key and value cache tensors have shape `(batch_size, num_heads, seq_len, head_dim)`.
            Convolution and ssm states tensors have shape `(batch_size, d_inner, d_conv)` and
            `(batch_size, d_inner, d_state)` respectively.
            See the `Zamba2HybridDynamicCache` class for more details.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Zamba2 Model outputting raw hidden-states without any specific head on top.",
    ZAMBA2_START_DOCSTRING,
)
class Zamba2Model(ZambaModel, Zamba2PreTrainedModel):
    """
    Model consisting of *config.num_hidden_layers* layers.

    Args:
        config: Zamba2Config
    """

    def __init__(self, config: Zamba2Config):
        Zamba2PreTrainedModel.__init__(self, config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        blocks = [Zamba2AttentionDecoderLayer(config, block_id=k) for k in range(config.num_mem_blocks)]
        mamba_layers = []
        linear_layers = []
        self.layers_block_type = config.layers_block_type
        for i in range(config.num_hidden_layers):
            if config.layers_block_type[i] == "mamba":
                mamba_layers.append(Zamba2MambaDecoderLayer(config, layer_idx=i))
            elif config.layers_block_type[i] == "hybrid":
                linear_layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False))
                mamba_layers.append(Zamba2MambaDecoderLayer(config, layer_idx=i))
        mamba_layers = iter(mamba_layers)
        linear_layers = iter(linear_layers)
        blocks = cycle(blocks)
        layers = self.get_layers(blocks, linear_layers, mamba_layers)
        self.layers = nn.ModuleList(layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_mem_rope:
            if config.use_long_context:
                logger.warning_once(
                    "`use_long_context` set to `True`: using rescaled `rope_theta` and extended `max_position_embeddings`."
                )
            self.rotary_emb = Zamba2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_layers(self, blocks, linear_layers, mamba_layers):
        layers = []
        self._tied_weights_keys = []
        self.first_transformer_layer_id = 0
        for layer_id, layer_type in enumerate(self.layers_block_type):
            if layer_type == "hybrid":
                if self.first_transformer_layer_id == 0:
                    self.first_transformer_layer_id = layer_id
                block = next(blocks)
                if self.config.num_mem_blocks * len(layer_type_list(self.config)) > 1:
                    prefix_name = f"layers.{layer_id}."
                    tied_keys = [
                        "shared_transformer.self_attn.q_proj.weight",
                        "shared_transformer.self_attn.k_proj.weight",
                        "shared_transformer.self_attn.v_proj.weight",
                        "shared_transformer.self_attn.o_proj.weight",
                        "shared_transformer.feed_forward.gate_up_proj.weight",
                        "shared_transformer.feed_forward.down_proj.weight",
                        "shared_transformer.input_layernorm.weight",
                        "shared_transformer.pre_ff_layernorm.weight",
                    ]
                    self._tied_weights_keys = [*self._tied_weights_keys, *[prefix_name + key for key in tied_keys]]
                    if self.config.use_shared_mlp_adapter:
                        tied_keys_adapter = []
                        adapter_id = 0
                        for _layer_type in self.layers_block_type:
                            if _layer_type == "hybrid" and adapter_id % self.config.num_mem_blocks == block.block_id:
                                tied_keys_adapter.append(
                                    "shared_transformer.feed_forward.gate_up_proj_adapter_list."
                                    + str(adapter_id)
                                    + ".0.weight"
                                )
                                tied_keys_adapter.append(
                                    "shared_transformer.feed_forward.gate_up_proj_adapter_list."
                                    + str(adapter_id)
                                    + ".1.weight"
                                )
                            adapter_id += 1
                        self._tied_weights_keys = [*self._tied_weights_keys, *tied_keys_adapter]
                    if self.config.use_shared_attention_adapter:
                        tied_keys_adapter = []
                        adapter_id = 0
                        for _layer_type in self.layers_block_type:
                            if _layer_type == "hybrid" and adapter_id % self.config.num_mem_blocks == block.block_id:
                                tied_keys_adapter.append(
                                    "shared_transformer.self_attn.linear_q_adapter_list."
                                    + str(adapter_id)
                                    + ".0.weight"
                                )
                                tied_keys_adapter.append(
                                    "shared_transformer.self_attn.linear_k_adapter_list."
                                    + str(adapter_id)
                                    + ".0.weight"
                                )
                                tied_keys_adapter.append(
                                    "shared_transformer.self_attn.linear_v_adapter_list."
                                    + str(adapter_id)
                                    + ".0.weight"
                                )
                                tied_keys_adapter.append(
                                    "shared_transformer.self_attn.linear_q_adapter_list."
                                    + str(adapter_id)
                                    + ".1.weight"
                                )
                                tied_keys_adapter.append(
                                    "shared_transformer.self_attn.linear_k_adapter_list."
                                    + str(adapter_id)
                                    + ".1.weight"
                                )
                                tied_keys_adapter.append(
                                    "shared_transformer.self_attn.linear_v_adapter_list."
                                    + str(adapter_id)
                                    + ".1.weight"
                                )
                            adapter_id += 1
                        self._tied_weights_keys = [*self._tied_weights_keys, *tied_keys_adapter]
                layers.append(Zamba2HybridLayer(block, next(linear_layers), next(mamba_layers)))
            else:
                layers.append(next(mamba_layers))
        return layers

    @add_start_docstrings_to_model_forward(ZAMBA2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Zamba2HybridDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        original_hidden_states = torch.clone(inputs_embeds)
        # original_hidden_states: word embedding output that will be concatenated with hidden activations to form the input of the shared transformer layer

        if use_cache and past_key_values is None:
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            past_key_values = Zamba2HybridDynamicCache(self.config, batch_size, dtype=self.dtype, device=self.device)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length(layer_idx=self.first_transformer_layer_id)
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # create position embeddings to be shared across the decoder layers
        if self.config.use_mem_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    original_hidden_states,
                    layer_idx,
                    attention_mask,
                    causal_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    original_hidden_states=original_hidden_states,
                    layer_idx=layer_idx,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    # append attentions only of attention layers. Mamba layers return `None` as the attention weights
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class Zamba2ForCausalLM(ZambaForCausalLM):
    pass


@add_start_docstrings(
    """
    The Zamba2 Model with a sequence classification head on top (linear layer).

    [`Zamba2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    ZAMBA2_START_DOCSTRING,
)
class Zamba2ForSequenceClassification(ZambaForSequenceClassification):
    pass


__all__ = [
    "Zamba2ForCausalLM",
    "Zamba2ForSequenceClassification",
    "Zamba2Model",
    "Zamba2PreTrainedModel",
]
