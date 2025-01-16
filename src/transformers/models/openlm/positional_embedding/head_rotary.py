# NOTE: 08/31/23, this class is copied from xformers as there is currently a bug related to which channel dim the rotary embedding is applied to.
# when the upstream issue is fixed, this file should be deleted. To track progress, see this issue: https://github.com/facebookresearch/xformers/issues/841

# taken from: https://github.com/facebookresearch/xformers/blob/748c159096d4f9fcfe3eaf22801e5aed4777210b/xformers/components/positional_embedding/rotary.py

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This implementation is inspired by GPT-NeoX https://github.com/EleutherAI/gpt-neox
# NOTE: Almost the same right now, moving parts to Triton is the next step

from typing import Tuple

import torch

from .rotary import apply_rotary_pos_emb, RotaryEmbedding


class HeadRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings used in the first version of OpenLM.
    It is only kept for compatibility, RotaryEmbedding should be used instead.
    """

    def __init__(self, dim_model: int, seq_len: int, *_, **__):
        super().__init__(dim_model, seq_len)
        self._has_warned = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_tables(k.shape[2], device=k.device, dtype=k.dtype)

        if not self._has_warned and (offset != 0):
            print("Warning. HeadRotaryEmbedding does not support offset, I am not applying it.")
            self._has_warned = True

        out_q = apply_rotary_pos_emb(q.transpose(1, 2), self._cos_cached, self._sin_cached).transpose(1, 2)
        out_k = apply_rotary_pos_emb(k.transpose(1, 2), self._cos_cached, self._sin_cached).transpose(1, 2)
        return out_q, out_k


class HeadRotaryWithCast(HeadRotaryEmbedding):
    # NOTE: this version has the bug, but we trained the 7B model with it so it's default
    def forward(self, q, k, v, offset: int = 0):
        q, k = super().forward(q, k, offset)
        return q.to(v.dtype), k.to(v.dtype), v