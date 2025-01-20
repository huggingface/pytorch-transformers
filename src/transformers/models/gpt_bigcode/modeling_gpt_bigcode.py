# coding=utf-8
# Copyright 2023 The Bigcode team and HuggingFace Inc. team.
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
"""PyTorch GPTBigCode model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_2_2
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_gpt_bigcode import GPTBigCodeConfig


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigcode/gpt_bigcode-santacoder"
_CONFIG_FOR_DOC = "GPTBigCodeConfig"


# Fused kernels
# Use separate functions for each case because conditionals prevent kernel fusion.
# TODO: Could have better fused kernels depending on scaling, dropout and head mask.
#  Is it doable without writing 32 functions?
@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


class GPTBigCodeAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config

        self.mask_value = None
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        self.is_causal = True

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )
        self.attn_pdrop = config.attn_pdrop

        if self.is_cross_attention:
            if self.multi_query:
                raise NotImplementedError("Multi-Query Attention not supported for cross_attention")

            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.kv_dim)

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)
        if self.multi_query:
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)
        else:
            # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
            # -> (batch_size, num_heads, query_length, key_length)
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            # Always copies
            query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == "cpu":
            # This is needed because of a bug in pytorch https://github.com/pytorch/pytorch/issues/80588.
            # The bug was fixed in https://github.com/pytorch/pytorch/pull/96086,
            # but the fix has not been released as of pytorch version 2.0.0.
            attn_weights = torch.zeros_like(attn_weights)
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(attn_shape)

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask

        if self.multi_query:
            attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)
        else:
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        attn_output, attn_weights = self._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)

        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTBigCodeFlashAttention2(GPTBigCodeAttention):
    """
    GPTBigCode flash attention module. This module inherits from `GPTBigCodeAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass where it needs to correctly call the public
    API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        if self.multi_query:
            batch_size, query_length, _ = query.shape
            query = query.reshape(batch_size, query_length, self.num_heads, self.head_dim)
            key = key.unsqueeze(2)
            value = value.unsqueeze(2)
        else:
            query_length = query.shape[2]
            batch_size, _, tgt, _ = key.shape
            query = query.transpose(1, 2).reshape(batch_size, query_length, self.num_heads, self.head_dim)
            key = key.transpose(1, 2).reshape(batch_size, tgt, self.num_heads, self.head_dim)
            value = value.transpose(1, 2).reshape(batch_size, tgt, self.num_heads, self.head_dim)

        attn_dropout = self.attn_pdrop if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.c_attn.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        attn_output = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length,
            dropout=attn_dropout,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_weights_reshaped = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        attn_output = self.c_proj(attn_weights_reshaped)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        if output_attentions:
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights_reshaped = attn_weights_reshaped.transpose(1, 2)
        else:
            attn_weights_reshaped = None

        outputs += (attn_weights_reshaped,)

        return outputs  # a, present, (attentions)


class GPTBigCodeSdpaAttention(GPTBigCodeAttention):
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if head_mask is not None:
            # The super dispatch is done in the forward.
            raise ValueError(
                "PyTorch SDPA does not support head_mask. Please open an issue in Transformers repository."
            )

        scale = None
        if not self.scale_attn_weights:
            scale = 1

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key.shape[-2]

        if self.multi_query:
            query_length = query_shape[1]

            # SDPA requires the dimension [..., sequence_length, head_dim].
            query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)

            # Without these unsqueeze, SDPA complains as the query and key/value have a different number of dimensions.
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

            # Although these expand are not numerically useful, PyTorch can not dispatch to memory-efficient backend
            # and flash attention backend (No available kernel.  Aborting execution.) from the shapes
            # query = [batch_size, num_heads, query_length, head_dim]
            # key = [batch_size, 1, past_length, head_dim]
            # value = [batch_size, 1, past_length, head_dim]
            #
            # torch==2.1.2 is bugged with non-contiguous inputs with custom attn_mask (https://github.com/pytorch/pytorch/issues/112577), hence the check.
            if is_torch_greater_or_equal_than_2_2:
                key = key.expand(-1, self.num_heads, -1, -1)
                value = value.expand(-1, self.num_heads, -1, -1)
        else:
            query_length = query_shape[-1]

            # See the comment above.
            if query.device.type == "cuda" and attention_mask is not None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not
        # create a causal mask in case query_length == 1.
        is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=is_causal,
            scale=scale,
        )

        if self.multi_query:
            # (batch_size, num_heads, seq_len, head_dim) --> (batch_size, seq_len, num_heads, head_dim)
            sdpa_result = sdpa_result.transpose(1, 2)

            # Reshape is kind of expensive here, as it does a memory copy,
            # but I did not manage to make away without it (logits do not match when using view)
            # (batch_size, seq_len, num_heads, head_dim) --> (batch_size, seq_len, num_heads * head_dim)
            sdpa_result = sdpa_result.reshape(query_shape)

        return sdpa_result, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        if not output_attentions and head_mask is None:
            # Difference with the original implementation: there is no need to transpose the key here,
            # as SDPA expects seq_length to be at index -2 for the key as well
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        else:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "GPTBigCodeModel is using GPTBigCodeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` and `head_mask` not None."
                ' Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            attn_output, attn_weights = super()._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)

        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        return outputs


class GPTBigCodeMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


GPTBIGCODE_ATTENTION_CLASSES = {
    "eager": GPTBigCodeAttention,
    "flash_attention_2": GPTBigCodeFlashAttention2,
    "sdpa": GPTBigCodeSdpaAttention,
}


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.attn = GPTBIGCODE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            if config.multi_query:
                raise NotImplementedError("Cross-attention not implemented for MQA")

            self.crossattention = GPTBIGCODE_ATTENTION_CLASSES[config._attn_implementation](
                config, is_cross_attention=True, layer_idx=layer_idx
            )

            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPTBigCodeMLP(self.inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTBigCodeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTBigCodeBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            )
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, nn.Linear):
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


GPT_BIGCODE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTBigCodeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_BIGCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[torch.Tensor]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.Tensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
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


@add_start_docstrings(
    "The bare GPT_BIGCODE Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)), persistent=False
        )

        self.gradient_checkpointing = False

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0].size(-2)

        if attention_mask is not None and len(attention_mask.shape) == 2 and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, past_length : input_shape[-1] + past_length :]
        elif position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Self-attention mask.
        query_length = input_shape[-1]
        key_length = past_length + query_length
        self_attention_mask = self.bias[None, key_length - query_length : key_length, :key_length]

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask.bool() if (attention_mask is not None and 0 in attention_mask) else None
            encoder_attention_mask = (
                encoder_attention_mask.bool()
                if (encoder_attention_mask is not None and 0 in encoder_attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is not None:
                self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
                    dtype=torch.bool, device=self_attention_mask.device
                )

            # MQA models: (batch_size, query_length, n_heads, key_length)
            # MHA models: (batch_size, n_heads, query_length, key_length)
            self_attention_mask = self_attention_mask.unsqueeze(2 if self.multi_query else 1)

            if self._use_sdpa and head_mask is None and not output_attentions:
                # SDPA with a custom mask is much faster in fp16/fp32 dtype rather than bool. Cast here to floating point instead of at every layer.
                dtype = self.wte.weight.dtype
                min_dtype = torch.finfo(dtype).min
                self_attention_mask = torch.where(
                    self_attention_mask,
                    torch.full([], 0.0, dtype=dtype, device=self_attention_mask.device),
                    torch.full([], min_dtype, dtype=dtype, device=self_attention_mask.device),
                )

                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                if self.multi_query:
                    # gpt_bigcode using MQA has the bad taste to use a causal mask with shape
                    # [batch_size, target_length, 1, source_length], not compatible with SDPA, hence this transpose.
                    self_attention_mask = self_attention_mask.transpose(1, 2)

                if query_length > 1 and attention_mask is not None and attention_mask.device.type in ["cuda", "xpu"]:
                    # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
                    # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
                    self_attention_mask = AttentionMaskConverter._unmask_unattended(
                        self_attention_mask, min_dtype=min_dtype
                    )

            attention_mask = self_attention_mask

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if (
                self.config.add_cross_attention
                and encoder_hidden_states is not None
                and encoder_attention_mask is not None
            ):
                if encoder_attention_mask.dim() == 2:
                    encoder_attention_mask.unsqueeze(1)
                assert encoder_attention_mask.dim() == 3
                encoder_attention_mask = encoder_attention_mask.bool().unsqueeze(2 if self.multi_query else 1)
            else:
                encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """
    The GPT_BIGCODE Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Overwritten -- `past_key_values` with uncommon shape

        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """
        Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length.
        Since gpt bigcode is special, the method is overridden here, other models use it from `generation.utils.py`.
        """
        past_length = 0
        if "past_key_values" in model_kwargs:
            if self.config.multi_query:
                past_length = model_kwargs["past_key_values"][0].shape[1]
            else:
                past_length = model_kwargs["past_key_values"][0].shape[2]
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        else:
            cur_len = input_ids.shape[-1]
        model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=input_ids.device)
        return model_kwargs

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)


@add_start_docstrings(
    """
    The GPTBigCode Model transformer with a sequence classification head on top (linear layer).

    [`GPTBigCodeForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForSequenceClassification(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTBigCodeModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
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
            labels = labels.to(logits.device)

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    GPT_BIGCODE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForTokenClassification(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPTBigCodeModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "GPTBigCodeForSequenceClassification",
    "GPTBigCodeForTokenClassification",
    "GPTBigCodeForCausalLM",
    "GPTBigCodeModel",
    "GPTBigCodePreTrainedModel",
]
