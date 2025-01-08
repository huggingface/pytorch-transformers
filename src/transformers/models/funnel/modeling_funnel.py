# coding=utf-8
# Copyright 2020-present Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
"""PyTorch Funnel Transformer model."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_funnel import FunnelConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FunnelConfig"
_CHECKPOINT_FOR_DOC = "funnel-transformer/small"


INF = 1e6


def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
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

    _layer_map = {
        "k": "k_head",
        "q": "q_head",
        "v": "v_head",
        "o": "post_proj",
        "layer_1": "linear_1",
        "layer_2": "linear_2",
        "rel_attn": "attention",
        "ff": "ffn",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
        "lookup_table": "weight",
        "word_embedding": "word_embeddings",
        "input": "embeddings",
    }

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        if name[0] == "generator":
            continue
        pointer = model
        skipped = False
        for m_name in name[1:]:
            if not isinstance(pointer, FunnelPositionwiseFFN) and re.fullmatch(r"layer_\d+", m_name):
                layer_index = int(re.search(r"layer_(\d+)", m_name).groups()[0])
                if layer_index < config.num_hidden_layers:
                    block_idx = 0
                    while layer_index >= config.block_sizes[block_idx]:
                        layer_index -= config.block_sizes[block_idx]
                        block_idx += 1
                    pointer = pointer.blocks[block_idx][layer_index]
                else:
                    layer_index -= config.num_hidden_layers
                    pointer = pointer.layers[layer_index]
            elif m_name == "r" and isinstance(pointer, FunnelRelMultiheadAttention):
                pointer = pointer.r_kernel
                break
            elif m_name in _layer_map:
                pointer = getattr(pointer, _layer_map[m_name])
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    print(f"Skipping {'/'.join(name)}", array.shape)
                    skipped = True
                    break
        if not skipped:
            if len(pointer.shape) != len(array.shape):
                array = array.reshape(pointer.shape)
            if m_name == "kernel":
                array = np.transpose(array)
            pointer.data = torch.from_numpy(array)

    return model


class FunnelEmbeddings(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        return embeddings


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # divided.
        self.pooling_mult = None

    def init_attention_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Returns the attention inputs associated to the inputs of the model."""
        # inputs_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.size(1)
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        cls_mask = (
            nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config.separate_cls
            else None
        )
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Convert `token_type_ids` to `token_type_mat`."""
        token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        # Treat <cls> as in the same segment as both A & B
        cls_ids = token_type_ids == self.cls_token_type_id
        cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        return cls_mat | token_type_mat

    def get_position_embeds(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> Union[Tuple[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        d_model = self.config.d_model
        if self.config.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula.
            # We need to create and return the matrices phi, psi, pi and omega.
            pos_seq = torch.arange(0, seq_len, 1.0, dtype=torch.int64, device=device).to(dtype)
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=torch.int64, device=device).to(dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            sinusoid = pos_seq[:, None] * inv_freq[None]
            sin_embed = torch.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = torch.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            # This is different from the formula on the paper...
            phi = torch.cat([sin_embed_d, sin_embed_d], dim=-1)
            psi = torch.cat([cos_embed, sin_embed], dim=-1)
            pi = torch.cat([cos_embed_d, cos_embed_d], dim=-1)
            omega = torch.cat([-sin_embed, cos_embed], dim=-1)
            return (phi, pi, psi, omega)
        else:
            # Notations from the paper, appending A.2.1, final formula.
            # We need to create and return all the possible vectors R for all blocks and shifts.
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=torch.int64, device=device).to(dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=torch.int64, device=device).to(dtype)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id[:, None] * inv_freq[None]
            sin_embed = self.sin_dropout(torch.sin(sinusoid))
            cos_embed = self.cos_dropout(torch.cos(sinusoid))
            pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)

            pos = torch.arange(0, seq_len, dtype=torch.int64, device=device).to(dtype)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config.num_blocks):
                # For each block with block_index > 0, we need two types position embeddings:
                #   - Attention(pooled-q, unpooled-kv)
                #   - Attention(pooled-q, pooled-kv)
                # For block_index = 0 we only need the second one and leave the first one as None.

                # First type
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    rel_pos = rel_pos[:, None] + zero_offset
                    rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                    position_embeds_pooling = torch.gather(pos_embed, 0, rel_pos)

                # Second type
                pos = pooled_pos
                stride = 2**block_index
                rel_pos = self.relative_pos(pos, stride)

                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                position_embeds_no_pooling = torch.gather(pos_embed, 0, rel_pos)

                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id: torch.Tensor, block_index: int):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        if self.config.separate_cls:
            # Under separate <cls>, we treat the <cls> as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will be at `1 - 2 ** block_index`.
            cls_pos = pos_id.new_tensor([-(2**block_index) + 1])
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos: torch.Tensor, stride: int, pooled_pos=None, shift: int = 1) -> torch.Tensor:
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos

        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)

    def stride_pool(
        self,
        tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        axis: Union[int, Tuple[int], List[int]],
    ) -> torch.Tensor:
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None

        # Do the stride pool recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # Do the stride pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # Deal with negative axis
        axis %= tensor.ndim

        axis_slice = (
            slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
        )
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)
        return tensor[enc_slice]

    def pool_tensor(
        self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], mode: str = "mean", stride: int = 2
    ) -> torch.Tensor:
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        if mode == "mean":
            tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(
        self, output, attention_inputs: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        else:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


def _relative_shift_gather(positional_attn: torch.Tensor, context_len: int, shift: int) -> torch.Tensor:
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    # max_rel_len = 2 * context_len + shift -1 is the numbers of possible relative positions i-j

    # What's next is the same as doing the following gather, which might be clearer code but less efficient.
    # idxs = context_len + torch.arange(0, context_len).unsqueeze(0) - torch.arange(0, seq_len).unsqueeze(1)
    # # matrix of context_len + i-j
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    positional_attn = torch.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    positional_attn = positional_attn[..., :context_len]
    return positional_attn


class FunnelRelMultiheadAttention(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int) -> None:
        super().__init__()
        self.config = config
        self.block_index = block_index
        d_model, n_head, d_head = config.d_model, config.n_head, config.d_head

        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_head = nn.Linear(d_model, n_head * d_head)
        self.v_head = nn.Linear(d_model, n_head * d_head)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_r_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_kernel = nn.Parameter(torch.zeros([d_model, n_head, d_head]))
        self.r_s_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.seg_embed = nn.Parameter(torch.zeros([2, n_head, d_head]))

        self.post_proj = nn.Linear(n_head * d_head, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.scale = 1.0 / (d_head**0.5)

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        if self.config.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            phi, pi, psi, omega = position_embeds
            # Shape n_head x d_head
            u = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape batch_size x sea_len x n_head x d_model
            q_r_attention = torch.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]

            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = torch.einsum("bind,jd->bnij", q_r_attention_1, psi) + torch.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            r = position_embeds[self.block_index][shift - 1]
            # Shape n_head x d_head
            v = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape max_rel_len x n_head x d_model
            r_head = torch.einsum("td,dnh->tnh", r, w_r)
            # Shape batch_size x n_head x seq_len x max_rel_len
            positional_attn = torch.einsum("binh,tnh->bnit", q_head + v, r_head)
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head has shape batch_size x seq_len x n_head x d_head
        # Shape n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # Shape batch_size x n_head x seq_len x 2
        token_type_bias = torch.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.shape[2], seq_len, context_len])
        # Shapes batch_size x n_head x seq_len
        diff_token_type, same_token_type = torch.split(token_type_bias, 1, dim=-1)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_attn = torch.where(
            token_type_mat, same_token_type.expand(token_type_mat.shape), diff_token_type.expand(token_type_mat.shape)
        )

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs: Tuple[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config.n_head, self.config.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, d_head)

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        attn_score = attn_score.float()
        # perform masking
        if attention_mask is not None:
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
        # attention probability
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
        attn_prob = self.attention_dropout(attn_prob)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(attn_vec.reshape(batch_size, seq_len, n_head * d_head))
        attn_out = self.hidden_dropout(attn_out)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class FunnelPositionwiseFFN(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.linear_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        return self.layer_norm(hidden + h)


class FunnelLayer(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int) -> None:
        super().__init__()
        self.attention = FunnelRelMultiheadAttention(config, block_index)
        self.ffn = FunnelPositionwiseFFN(config)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs,
        output_attentions: bool = False,
    ) -> Tuple:
        attn = self.attention(query, key, value, attention_inputs, output_attentions=output_attentions)
        output = self.ffn(attn[0])
        return (output, attn[1]) if output_attentions else (output,)


class FunnelEncoder(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for layer_index, layer in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


def upsample(
    x: torch.Tensor, stride: int, target_len: int, separate_cls: bool = True, truncate_seq: bool = False
) -> torch.Tensor:
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    output = torch.repeat_interleave(x, repeats=stride, dim=1)
    if separate_cls:
        if truncate_seq:
            output = nn.functional.pad(output, (0, 0, 0, stride - 1, 0, 0))
        output = output[:, : target_len - 1]
        output = torch.cat([cls, output], dim=1)
    else:
        output = output[:, :target_len]
    return output


class FunnelDecoder(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.layers = nn.ModuleList([FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    def forward(
        self,
        final_hidden: torch.Tensor,
        first_block_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        upsampled_hidden = upsample(
            final_hidden,
            stride=2 ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            separate_cls=self.config.separate_cls,
            truncate_seq=self.config.truncate_seq,
        )

        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        for layer in self.layers:
            layer_output = layer(hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions)
            hidden = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dense_prediction = nn.Linear(config.d_model, 1)

    def forward(self, discriminator_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits


class FunnelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FunnelConfig
    load_tf_weights = load_tf_weights_in_funnel
    base_model_prefix = "funnel"

    def _init_weights(self, module):
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                nn.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)
        elif classname == "FunnelRelMultiheadAttention":
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.r_r_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.r_kernel, b=self.config.initializer_range)
            nn.init.uniform_(module.r_s_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.seg_embed, b=self.config.initializer_range)
        elif classname == "FunnelEmbeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            nn.init.normal_(module.word_embeddings.weight, std=std)
            if module.word_embeddings.padding_idx is not None:
                module.word_embeddings.weight.data[module.word_embeddings.padding_idx].zero_()


class FunnelClassificationHead(nn.Module):
    def __init__(self, config: FunnelConfig, n_labels: int) -> None:
        super().__init__()
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.linear_out = nn.Linear(config.d_model, n_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_hidden(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.dropout(hidden)
        return self.linear_out(hidden)


@dataclass
class FunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FunnelForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss of the ELECTRA-style objective.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
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
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


FUNNEL_START_DOCSTRING = r"""

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FUNNEL_INPUTS_DOCSTRING = r"""
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
    """
    The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)

        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        inputs_embeds = self.embeddings(input_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


@add_start_docstrings(
    "The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.",
    FUNNEL_START_DOCSTRING,
)
class FunnelModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)
        self.decoder = FunnelDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        inputs_embeds = self.embeddings(input_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[1][self.config.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        return BaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


add_start_docstrings(
    """
    Funnel Transformer model with a binary classification head on top as used during pretraining for identifying
    generated tokens.
    """,
    FUNNEL_START_DOCSTRING,
)


class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FunnelForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the ELECTRA-style loss. Input should be a sequence of tokens (see `input_ids`
            docstring) Indices should be in `[0, 1]`:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FunnelForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")
        >>> model = FunnelForPreTraining.from_pretrained("funnel-transformer/small")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> logits = model(**inputs).logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return FunnelForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


@add_start_docstrings("""Funnel Transformer Model with a `language modeling` head on top.""", FUNNEL_START_DOCSTRING)
class FunnelForMaskedLM(FunnelPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        prediction_logits = self.lm_head(last_hidden_state)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Funnel Transformer Model with a sequence classification/regression head on top (two linear layer on top of the
    first timestep of the last hidden state) e.g. for GLUE tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForSequenceClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Funnel Transformer Model with a multiple choice classification head on top (two linear layer on top of the first
    timestep of the last hidden state, and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForMultipleChoice(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)

        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, 1)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Funnel Transformer Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForTokenClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels

        self.funnel = FunnelModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)

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
    Funnel Transformer Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForQuestionAnswering(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels

        self.funnel = FunnelModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeze(-1)
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
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "FunnelBaseModel",
    "FunnelForMaskedLM",
    "FunnelForMultipleChoice",
    "FunnelForPreTraining",
    "FunnelForQuestionAnswering",
    "FunnelForSequenceClassification",
    "FunnelForTokenClassification",
    "FunnelModel",
    "FunnelPreTrainedModel",
    "load_tf_weights_in_funnel",
]
