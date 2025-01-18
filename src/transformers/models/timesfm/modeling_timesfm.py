# coding=utf-8
# Copyright 2024 Google LLC and HuggingFace Inc. team.
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
"""PyTorch TimesFM model."""

import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_timesfm import TimesFmConfig


@dataclass
class TimesFmDecoderOutput(BaseModelOutput):
    loc: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None


@dataclass
class TimesFmOutputForPrediction(BaseModelOutput):
    mean_predictions: Optional[torch.Tensor] = None
    full_predictions: Optional[torch.Tensor] = None
    loss: Optional[Union[torch.Tensor, float]] = None


class TimesFmTransformerMLP(nn.Module):
    """Pax transformer MLP in pytorch."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

    def forward(self, x, paddings=None):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFmResidualBlock(nn.Module):
    """TimesFM residual block."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Hidden Layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        # Residual Layer
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class TimesFmRMSNorm(torch.nn.Module):
    """Pax rms norm in pytorch."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class TimesFmPositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence.

    Attributes:
        embedding_dims: Dimension of the embedding to be generated.
        min_timescale: Start of the geometric index. Determines the periodicity of
          the added signal. Defaults to 1.
        max_timescale: End of the geometric index. Determines the frequency of the
          added signal. Defaults to 10_000.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
    ) -> None:
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims

    def forward(self, seq_length=None, position=None):
        """Generates a Tensor of sinusoids with different frequencies.

        Args:
            seq_length: an optional Python int defining the output sequence length.
              if the `position` argument is specified.
            position:   [B, seq_length], optional position for each token in the
              sequence, only required when the sequence is packed.

        Returns:
            [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None:
            assert seq_length is not None
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            assert position.ndim == 2, position.shape

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(float(self.max_timescale) / float(self.min_timescale)) / max(
            num_timescales - 1, 1
        )
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        # Padding to ensure correct embedding dimension
        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal


class TimesFmAttention(nn.Module):
    """Implements the attention used in TimesFM. One key difference is that there is _per_dim_scaling of the query."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()
        self.attention_dropout = config.attention_dropout

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.model_dim
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = nn.Parameter(torch.empty((self.head_dim,)))

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def _scale_query(self, query: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.scaling).mul(1.442695041 / math.sqrt(self.head_dim))
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_write_indices: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xq = self._scale_query(xq)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        if kv_cache is not None and kv_write_indices is not None:
            k_cache, v_cache = kv_cache
            k_cache.index_copy_(1, kv_write_indices, xk)
            v_cache.index_copy_(1, kv_write_indices, xv)

            key = k_cache
            value = v_cache
        else:
            key = xk
            value = xv
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3))

        if attention_mask is not None:
            scores = scores + attention_mask

        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)
        # return scores, output.transpose(1, 2).contiguous()

        # [batch_size, input_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)

        if not output_attentions:
            scores = None

        return output, scores


class TimesFmSdpaAttention(TimesFmAttention):
    """TimesFM attention implementation using torch.nn.functional.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_write_indices: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )

        hidden_states_shape = hidden_states.shape
        batch_size, seq_length, _ = hidden_states_shape

        # Project to queries, keys, values
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape: [batch_size, seq_length, num_heads * head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        # Scale query exactly as in original
        xq = self._scale_query(xq)

        # Handle KV cache
        if kv_cache is not None and kv_write_indices is not None:
            k_cache, v_cache = kv_cache
            k_cache.index_copy_(1, kv_write_indices, xk)
            v_cache.index_copy_(1, kv_write_indices, xv)
            key = k_cache
            value = v_cache
        else:
            key = xk
            value = xv

        # Handle grouped attention
        if self.num_queries_per_kv > 1:
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # Transpose for attention: [batch_size, num_heads, seq_length, head_dim]
        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Make inputs contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Run scaled dot-product attention
        # Note: attention_mask should already be in the correct format from TimesFmStackedDecoder
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,  # We use the provided attention mask
            scale=1,  # We already scaled the query
        )

        # Reshape output: [batch_size, seq_length, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


TIMESFM_ATTENTION_CLASSES = {
    "eager": TimesFmAttention,
    "sdpa": TimesFmSdpaAttention,
}


class TimesFmDecoderLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()

        if config._attn_implementation not in TIMESFM_ATTENTION_CLASSES:
            raise ValueError(f"Unknown attention implementation: {config._attn_implementation}")
        attention_class = TIMESFM_ATTENTION_CLASSES[config._attn_implementation]

        self.self_attn = attention_class(config)
        self.mlp = TimesFmTransformerMLP(config.model_dim, config.intermediate_size)
        self.input_layernorm = TimesFmRMSNorm(config.model_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        kv_write_indices: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, scores = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states, paddings=paddings)

        return scores, hidden_states


class TimesFmStackedDecoder(nn.Module):
    """Stacked transformer layer."""

    def __init__(self, config: TimesFmConfig):
        super().__init__()

        self.layers = nn.ModuleList([TimesFmDecoderLayer(config) for _ in range(config.num_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        paddings: torch.Tensor,
        kv_write_indices: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        padding_mask = timesfm_convert_paddings_to_mask(paddings, hidden_states.dtype)
        atten_mask = timesfm_causal_mask(hidden_states)
        mask = timesfm_merge_masks(padding_mask, atten_mask)
        all_attentions = []
        all_hidden_states = []

        for i in range(len(self.layers)):
            layer = self.layers[i]
            kv_cache = kv_caches[i] if kv_caches is not None else None
            scores, hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=mask,
                paddings=paddings,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions.append(scores)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_attentions,
            hidden_states=all_hidden_states,
        )


# Move utility functions here
def timesfm_masked_mean_std(inputs: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and standard deviation of `inputs` across axis 1.

    It excludes values where `padding` is 1.

    Args:
        inputs: A PyTorch tensor of shape [b, n, p].
        padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

    Returns:
        A tuple containing the mean and standard deviation.
        We return the statistics of the first patch with more than three non-padded values.
    """

    # Selecting the first patch with more than 3 unpadded values.
    def _get_patch_index(arr: torch.Tensor):
        indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
        row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    pad_sum = torch.sum(1 - padding, dim=2)
    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]

    # Create a mask where padding is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = torch.sum(mask, dim=1)
    num_valid_elements = torch.where(
        num_valid_elements == 0,
        torch.tensor(1, dtype=num_valid_elements.dtype, device=num_valid_elements.device),
        num_valid_elements,
    )

    # Calculate the masked sum and squared sum
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_squared_sum = torch.sum((arr * mask) ** 2, dim=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = torch.where(
        masked_var < 0.0,
        torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
        masked_var,
    )
    masked_std = torch.sqrt(masked_var)

    return masked_mean, masked_std


def timesfm_shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Shifts rows of seq based on the first 0 in each row of the mask.

    Args:
        mask: mask tensor of shape [B, N]
        seq: seq tensor of shape [B, N, P]

    Returns:
        The shifted sequence.
    """
    batch_size, num_seq, feature_dim = seq.shape

    new_mask: torch.BoolTensor = mask == 0

    # Use argmax to find the first True value in each row
    indices = new_mask.to(torch.int32).argmax(dim=1)

    # Handle rows with all zeros
    indices[~new_mask.any(dim=1)] = -1

    # Create index ranges for each sequence in the batch
    idx_range = torch.arange(num_seq).to(seq.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, feature_dim)

    # Calculate shifted indices for each element in each sequence
    shifted_idx = (idx_range - indices[:, None, None]) % num_seq

    # Gather values from seq using shifted indices
    shifted_seq = seq.gather(1, shifted_idx)

    return shifted_seq


def timesfm_moving_average(arr: torch.Tensor, window_size: int) -> list[torch.Tensor]:
    """Calculates the moving average using PyTorch's convolution function."""
    # Pad with zeros to handle initial window positions
    arr_padded = F.pad(arr, (window_size - 1, 0), "constant", 0)
    # Create a convolution kernel
    kernel = torch.ones(window_size, dtype=arr.dtype, device=arr.device) / window_size
    # Apply convolution to calculate the moving average
    smoothed_arr = F.conv1d(arr_padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    return [smoothed_arr, arr - smoothed_arr]


def timesfm_get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    """Returns a large negative value for the given dtype."""
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def timesfm_causal_mask(input_t: torch.Tensor) -> torch.Tensor:
    """Computes and returns causal mask.

    Args:
        input_t: A torch.Tensor of shape [B, T, D].

    Returns:
        An attention_mask torch.Tensor of shape [1, 1, T, T]. Attention mask has
        already been converted to large negative values.
    """
    assert input_t.dtype.is_floating_point, input_t.dtype
    large_negative_number = timesfm_get_large_negative_number(input_t.dtype)
    t = input_t.shape[1]
    col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
    row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
    mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
    return mask.unsqueeze(0).unsqueeze(0).to(input_t.device)  # Equivalent to jnp.newaxis


def timesfm_convert_paddings_to_mask(paddings: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Converts binary paddings to a logit mask ready to add to attention matrix.

    Args:
        paddings: binary torch.Tensor of shape [B, T], with 1 denoting padding
          token.
        dtype: data type of the input.

    Returns:
        A torch.Tensor of shape [B, 1, 1, T] ready to add to attention logits.
    """
    attention_mask = paddings.detach().clone()
    attention_mask = attention_mask[:, None, None, :]  # Equivalent to jnp.newaxis
    attention_mask *= timesfm_get_large_negative_number(dtype)
    return attention_mask


def timesfm_merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Merges 2 masks.

    logscale mask is expected but 0/1 mask is also fine.

    Args:
        a: torch.Tensor of shape [1|B, 1, 1|T, S].
        b: torch.Tensor of shape [1|B, 1, 1|T, S].

    Returns:
        torch.Tensor of shape [1|B, 1, 1|T, S].
    """

    def expand_t(key_mask):
        query_mask = key_mask.transpose(-1, -2)  # Equivalent of jnp.transpose
        return torch.minimum(query_mask, key_mask)

    if a.shape[2] != b.shape[2]:
        if a.shape[2] == 1:
            a = expand_t(a)
        else:
            assert b.shape[2] == 1
            b = expand_t(b)

    assert a.shape[1:] == b.shape[1:], f"a.shape={a.shape}, b.shape={b.shape}."
    return torch.minimum(a, b)  # Element-wise minimum, similar to jnp.minimum


class TimesFmPreTrainedModel(PreTrainedModel):
    """handles the loading for all models."""

    config_class = TimesFmConfig
    base_model_prefix = "timesfm"
    main_input_name = "inputs"
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=self.config.initializer_factor)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, TimesFmRMSNorm):
            nn.init.zeros_(module.weight)

        elif isinstance(module, TimesFmTransformerMLP):
            # Initialize gate projection
            module.gate_proj.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.gate_proj.bias is not None:
                nn.init.zeros_(module.gate_proj.bias)

            # Initialize down projection
            module.down_proj.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.down_proj.bias is not None:
                nn.init.zeros_(module.down_proj.bias)

            # Initialize layer norm
            nn.init.ones_(module.layer_norm.weight)
            nn.init.zeros_(module.layer_norm.bias)

        elif isinstance(module, TimesFmAttention):
            # Initialize qkv projection
            module.qkv_proj.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.qkv_proj.bias is not None:
                nn.init.zeros_(module.qkv_proj.bias)

            # Initialize output projection
            module.o_proj.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.o_proj.bias is not None:
                nn.init.zeros_(module.o_proj.bias)

            # Initialize scaling parameter
            nn.init.ones_(module.scaling)

        elif isinstance(module, TimesFmResidualBlock):
            # Initialize hidden layer
            module.hidden_layer[0].weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.hidden_layer[0].bias is not None:
                nn.init.zeros_(module.hidden_layer[0].bias)

            # Initialize output layer
            module.output_layer.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.output_layer.bias is not None:
                nn.init.zeros_(module.output_layer.bias)

            # Initialize residual layer
            module.residual_layer.weight.data.normal_(mean=0, std=self.config.initializer_factor)
            if module.residual_layer.bias is not None:
                nn.init.zeros_(module.residual_layer.bias)

        elif isinstance(module, TimesFmPositionalEmbedding):
            pass


class TimesFmDecoder(TimesFmPreTrainedModel):
    """Patched time-series decoder without any specific output layer."""

    def __init__(self, config: TimesFmConfig):
        super().__init__(config)

        self.config = config
        self.input_ff_layer = TimesFmResidualBlock(
            input_dims=2 * config.patch_len,
            output_dims=config.model_dim,
            hidden_dims=config.intermediate_size,
        )
        self.freq_emb = nn.Embedding(num_embeddings=config.freq_size, embedding_dim=config.model_dim)
        self.stacked_transformer = TimesFmStackedDecoder(config=config)
        if self.config.use_positional_embedding:
            self.position_emb = TimesFmPositionalEmbedding(
                embedding_dims=self.config.model_dim,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Input is of shape [B, N, P]."""
        mu, sigma = timesfm_masked_mean_std(inputs, patched_pads)
        sigma = torch.where(
            sigma < self.config.tolerance,
            torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
            sigma,
        )

        # Normalize each patch
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(self.config.pad_val, dtype=outputs.dtype, device=outputs.device),
            outputs,
        )
        return outputs, (mu, sigma)

    def _preprocess_input(
        self, input_ts: torch.Tensor, input_padding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Preprocess input for stacked transformer."""
        # Reshape into patches (using view for efficiency)
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
        patched_pads = input_padding.view(bsize, -1, self.config.patch_len)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1)[0]  # Get the values from the min result
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = timesfm_shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb

        return model_input, patched_padding, stats, patched_inputs

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[TimesFmDecoderOutput, tuple[torch.Tensor, ...]]:
        model_input, patched_padding, stats, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb

        transformer_output = self.stacked_transformer(
            model_input,
            patched_padding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            all_hidden_states = [model_input] + transformer_output.hidden_states
        else:
            all_hidden_states = None

        if return_dict:
            return TimesFmDecoderOutput(
                last_hidden_state=transformer_output.last_hidden_state,
                hidden_states=all_hidden_states,
                attentions=transformer_output.attentions if output_attentions else None,
                loc=stats[0],
                scale=stats[1],
            )
        else:
            return (
                transformer_output.last_hidden_state,
                all_hidden_states,
                transformer_output.attentions,
                stats[0],
                stats[1],
            )


class TimesFmModelForPrediction(TimesFmPreTrainedModel):
    """TimesFM model for quantile and mean prediction."""

    def __init__(self, config: TimesFmConfig):
        super().__init__(config)

        self.config = config
        self.context_len = config.context_len
        self.horizon_len = config.horizon_len

        self.decoder = TimesFmDecoder(config)

        # quantile and mean output
        self.horizon_ff_layer = TimesFmResidualBlock(
            input_dims=config.model_dim,
            output_dims=config.horizon_len * (1 + len(config.quantiles)),
            hidden_dims=config.intermediate_size,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _preprocess(
        self, inputs: Sequence[torch.Tensor], freq: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
          inputs: A list of 1d Tensors. Each Tensor is the context time series of
            a single forecast task.
          freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """
        input_ts, input_padding, inp_freq = [], [], []

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = torch.zeros(input_len + self.horizon_len, dtype=ts.dtype, device=ts.device)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = torch.cat([torch.zeros(num_front_pad, dtype=ts.dtype, device=ts.device), ts], dim=0)
                padding = torch.cat([torch.ones(num_front_pad, dtype=ts.dtype, device=padding.device), padding], dim=0)
            elif input_len > self.context_len:
                ts = ts[-self.context_len :]
                padding = padding[-(self.context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        return (
            torch.stack(input_ts, dim=0),
            torch.stack(input_padding, dim=0),
            torch.tensor(inp_freq, dtype=torch.int32).reshape(-1, 1),
        )

    def _postprocess_output(
        self, model_output: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Postprocess output of stacked transformer."""

        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)

        # Reshape using view
        b, n, _ = output_ts.shape
        output_ts = output_ts.view(b, n, self.config.horizon_len, len(self.config.quantiles) + 1)

        return self._reverse_transform(output_ts, stats)

    def _reverse_transform(self, outputs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Output is of shape [B, N, P, Q]."""
        mu, sigma = stats
        return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.config.quantiles):
            errors = targets - predictions[..., i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        freq: Optional[Sequence[Union[torch.Tensor,int]]] = None,
        window_size: Optional[int] = None,
        future_target: Optional[torch.Tensor] = None,
        forecast_context_len: Optional[int] = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TimesFmOutputForPrediction, tuple[torch.Tensor, ...]]:
        """Forecasts on a list of time series.

        Args:
          inputs: list of time series forecast contexts. Each context time series
            should be a torch Tensor of potentially different context lengths.
          freq: frequency of each context time series in the inputs. 0 for high frequency
            (default), 1 for medium, and 2 for low.
          window_size: window size of trend + residual decomposition. If None then
            we do not do decomposition.
          future_target: optional future target time series to be used for loss computation.
          forecast_context_len: optional max context length.
          return_forecast_on_context: True to return the forecast on the context
            when available, i.e. after the first input patch.
          truncate_negative: truncate to only non-negative values if all the contexts
            have non-negative values.
          output_attentions: Whether to return the attentions.
          output_hidden_states: Whether to return the hidden states.
          return_dict: Whether to return a TimesFmOutputForPrediction object.

        Returns:
          A TimesFmOutputForPrediction object containing:
          - the mean forecast of size (# inputs, # forecast horizon),
          - the full forecast (mean + quantiles) of size
            (# inputs,  # forecast horizon, 1 + # quantiles).
          - loss: the mean squared error loss + quantile loss if future_target is provided.
        """
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len

        # Get device from first input tensor
        device = inputs[0].device

        # Truncate inputs to forecast_context_len
        inputs = [ts[-fcontext_len:] for ts in inputs]
        inp_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

        if window_size is not None:
            new_inputs = []
            if freq is not None:
                new_freqs = []
            for i, ts in enumerate(inputs):
                new_inputs.extend(timesfm_moving_average(ts, window_size))
                if freq is not None:
                    new_freqs.extend([freq[i]] * 2)
            inputs = new_inputs
            if freq is not None:
                freq = new_freqs

        if freq is None:
            logging.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        input_ts, input_padding, inp_freq = self._preprocess(inputs, freq)

        # Move tensors to the same device as input
        input_ts = input_ts.to(device)
        input_padding = input_padding.to(device)
        inp_freq = inp_freq.to(device)
        
        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []

        if input_padding.shape[1] != final_out.shape[1] + self.horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {input_padding.shape[1]} != {final_out.shape[1]} + {self.horizon_len}"
            )
        output_patch_len = self.config.horizon_len

        num_decode_patches = (self.horizon_len + output_patch_len - 1) // output_patch_len
        for step_index in range(num_decode_patches):
            current_padding = input_padding[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -fcontext_len:]
            input_padding = current_padding[:, -fcontext_len:]
            decoder_output = self.decoder(
                input_ts,
                input_padding,
                inp_freq,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            fprop_outputs = self._postprocess_output(
                decoder_output.last_hidden_state,
                (decoder_output.loc, decoder_output.scale),
            )

            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.config.patch_len, :]
                # We have to use reshape and not view for non-contiguous memory
                new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1, new_full_ts.size(3))

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.config.patch_len + self.horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.concatenate(full_outputs, axis=1)[:, 0:self.horizon_len, :]

        mean_outputs = full_outputs[:, :, 0]
        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = torch.maximum(mean_outputs, 0.0)
            full_outputs = torch.maximum(full_outputs, 0.0)

        loss = None
        if future_target is not None:
            mse_loss = F.mse_loss(mean_outputs, future_target)
            quantile_loss = self._quantile_loss(full_outputs[:, :, 1:], future_target)
            loss = mse_loss + quantile_loss

        if return_dict:
            return TimesFmOutputForPrediction(
                last_hidden_state=decoder_output.last_hidden_state,
                attentions=decoder_output.all_attentions if output_attentions else None,
                hidden_states=decoder_output.all_hidden_states if output_hidden_states else None,
                mean_predictions=mean_outputs,
                full_predictions=full_outputs,
                loss=loss,
            )
        else:
            return_tuple = [decoder_output.last_hidden_state]
            if output_hidden_states:
                return_tuple.append(decoder_output.all_hidden_states)
            if output_attentions:
                return_tuple.append(decoder_output.all_attentions)
            return_tuple += [mean_outputs, full_outputs, loss]
            return tuple(return_tuple)
