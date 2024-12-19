from typing import Optional, Tuple

import torch

from ..modeling_flash_attention_utils import _flash_attention_forward
from ..utils import is_flash_attn_greater_or_equal_2_10


_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if query.dtype == torch.float32:
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        seq_len,
        module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        **kwargs,
    )

    return attn_output, None
