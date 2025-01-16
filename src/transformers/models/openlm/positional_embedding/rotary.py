# NOTE: 08/31/23, this class is copied from xformers as there is currently a bug related to which channel dim the rotary embedding is applied to.
# when the upstream issue is fixed, this file should be deleted. To track progress, see this issue: https://github.com/facebookresearch/xformers/issues/841

# taken from: https://github.com/facebookresearch/xformers/blob/748c159096d4f9fcfe3eaf22801e5aed4777210b/xformers/components/positional_embedding/rotary.py
from typing import Tuple

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, offset: int = 0):
    # NOTE: This could probably be moved to Triton
    assert (
        cos.shape[1] >= offset + x.shape[1]
    ), f"Offset and/or input sequence is too large,\
        \n offset: {offset}, seq_len: {x.shape[1]}, max: {cos.shape[1]}"

    # Handle a possible sequence length mismatch in between q and k
    cos_out = cos[:, offset : offset + x.shape[1], :, :]
    sin_out = sin[:, offset : offset + x.shape[1], :, :]

    return (x * cos_out) + (rotate_half(x) * sin_out)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, seq_len: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim_model = dim_model
        self.register_buffer("inv_freq", torch.zeros(self.dim_model // 2))

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self.seq_len = seq_len
        self.reset_parameters()

    def reset_parameters(self):
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_model, 2).float() / self.dim_model))
        self._update_cos_sin_tables(self.seq_len)

    def _update_cos_sin_tables(self, seq_len: int = None, device: torch.device = None, dtype: torch.dtype = None):
        # If no seq_len is provided, use the cached one
        # If the seq_len is smaller than the cached one it is included in the cached one so no need to update
        if seq_len is None or seq_len < self._seq_len_cached:
            seq_len = self._seq_len_cached

        # Reset the tables if the sequence length has increased,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            self._cos_cached = emb.cos()[None, :, None, :].to(dtype)
            self._sin_cached = emb.sin()[None, :, None, :].to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_tables(k.shape[1] + offset, device=k.device, dtype=k.dtype)
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, offset),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, offset),
        )


class RotaryWithCast(RotaryEmbedding):
    def forward(self, q, k, v, offset: int = 0):
        q, k = super().forward(q, k, offset)
        return q.to(v.dtype), k.to(v.dtype), v