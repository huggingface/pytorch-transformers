from functools import partial

import torch
from torch.nn import functional as F

try:
    import xformers.ops as xops
except ModuleNotFoundError:
    pass


def get_rectangular_causal_mask(shape, q_seq_len, k_seq_len, device, dtype):
    """Create a rectangular causal mask.

    This is especially useful when query length < key length, and ensures that the attention tensor comes from a tensor
    that initially has dimensions that are a multiple of 8, as required by xformers.

    >>> get_rectangular_causal_mask((1, 1), 2, 2, "cpu", torch.float32)
    tensor([[[[0., -inf],
              [0., 0.]]]])
    >>> get_rectangular_causal_mask((1, 1), 3, 5, "cpu", torch.float32)
    tensor([[[[0., 0., 0., -inf, -inf],
              [0., 0., 0., 0., -inf],
              [0., 0., 0., 0., 0.]]]])
    >>> get_rectangular_causal_mask((1, 1), 5, 5, "cpu", torch.float32)
    tensor([[[[0., -inf, -inf, -inf, -inf],
              [0., 0., -inf, -inf, -inf],
              [0., 0., 0., -inf, -inf],
              [0., 0., 0., 0., -inf],
              [0., 0., 0., 0., 0.]]]])
    """
    # xformers requires the mask to be built with a shape that is a multiple of 8
    next_multiple_8 = (k_seq_len + 7) // 8 * 8  #

    mask = torch.ones((q_seq_len, k_seq_len), device=device, dtype=bool)
    mask[:, -q_seq_len:] = torch.tril(mask[:, -q_seq_len:], diagonal=0)

    output_mask = torch.zeros((*shape, q_seq_len, next_multiple_8), device=device, dtype=dtype)
    output_mask[:, :, :, :k_seq_len].masked_fill_(~mask, torch.finfo(dtype).min)
    return output_mask[:, :, :, :k_seq_len]


def apply_attention_mask_(bias, attention_mask, queries_dtype):
    """Applies attention mask (e.g., from HuggingFace generate) to an attention bias mask in-place.

    Args:
        bias (torch.Tensor, shape (batch_size, num_heads, q_seq_len, k_seq_len))
        attention_mask (torch.Tensor, shape (batch_size, sequence_len))
        queries_dtype: queries.dtype; used to get minimum value for masked indices.

    Returns:
        bias_with_mask (torch.Tensor, shape (batch_size, num_heads, q_seq_len, k_seq_len))
    """
    # Update mask to remove attention based on attention_mask that's passed in.
    assert attention_mask.dim() == 2
    # From https://github.com/huggingface/transformers/blob/f738ab3b5d30e30c43a4c3d00ca8939f8a4d4427/src/transformers/models/llama/modeling_llama.py#L1089C1-L1091C117
    mask_length = attention_mask.shape[-1]
    # Set parts of bias that are zero (i.e., where attention is allowed) _and_ attention_mask is False (i.e.,
    # where we should not attend) with min_dtype.
    padding_mask = bias[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    min_dtype = torch.finfo(queries_dtype).min
    bias[..., :mask_length] = bias[..., :mask_length].masked_fill(padding_mask, min_dtype)
    # Disable masking for sequence indices where all attention weights are -inf
    # We won't use these anyway, and keeping them as -inf leads to nans.
    # See https://github.com/huggingface/transformers/blob/f738ab3b5d30e30c43a4c3d00ca8939f8a4d4427/src/transformers/modeling_attn_mask_utils.py#L189
    # for details.
    bias.mul_(~torch.all(bias == min_dtype, dim=-1, keepdim=True))


def xformers_attn(queries, keys, values, is_causal, attention_mask=None):
    # xformers assumes q, k, v are [batch, seq_len, heads, embed_dim]
    # We assume that queries match the last part of the key / value sequences
    # see (https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.LowerTriangularFromBottomRightMask)
    # we would like to replace the mask generation with: mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    # sadly we cannot us this because it needs xformers>=0.0.23 and this is not compatible with torch<2.1.1 while llm-foundry requires torch<2.1.1

    # If queries have shape [batch, 1, heads, dim] it means there is only one query in the sequence.
    # In this case, there is no notion of causal masking, so we can just set the mask to None.
    # This is actually needed to get the desired behavior with seq_len=1.
    bias = None
    if is_causal and queries.shape[1] == keys.shape[1] and attention_mask is None:
        bias = xops.LowerTriangularMask()
    elif is_causal and (queries.shape[1] > 1 or attention_mask is not None):
        # Build causal mask that assumes queries are in the end of the sequence.
        batch, q_seq_len, heads, _ = queries.shape
        k_seq_len = keys.shape[1]
        bias = get_rectangular_causal_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)
        if attention_mask is not None:
            apply_attention_mask_(bias, attention_mask, queries_dtype=queries.dtype)
    elif not is_causal and attention_mask is not None:
        raise NotImplementedError("attention_mask with is_causal=False is not yet implemented.")
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=bias)


def torch_attn(queries, keys, values, is_causal, attention_mask=None):
    # Need to call contiguous in torch >=2.1, otherwise later calls to .view() fail.
    # Possibly related: https://github.com/pytorch/pytorch/issues/110213 - behavior of scaled_dot_product_attention
    # changed between 2.0 and 2.1
    if is_causal and keys.shape[1] > queries.shape[1] > 1:
        q_seq_len = queries.shape[1]
        k_seq_len = keys.shape[1]
        # Same as above, we would like to use:
        # mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize((1, 1, q_seq_len, k_seq_len), queries.dtype, queries.device)
        mask = get_rectangular_causal_mask((1, 1), q_seq_len, k_seq_len, queries.device, queries.dtype)
        if attention_mask is not None:
            apply_attention_mask_(mask, attention_mask, queries_dtype=queries.dtype)
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
            )
            .transpose(1, 2)
            .contiguous()
        )
    else:
        if attention_mask is None:
            bias = None
            # If we only have one query, assume we don't need to be in causal mode (can attend to all keys).
            if queries.shape[1] == 1:
                is_causal = False
        else:
            if not is_causal:
                raise NotImplementedError("attention_mask with is_causal=False is not yet implemented.")
            # Build causal mask that assumes queries are in the end of the sequence.
            batch, q_seq_len, heads, _ = queries.shape
            k_seq_len = keys.shape[1]
            bias = get_rectangular_causal_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)
            if attention_mask is not None:
                apply_attention_mask_(bias, attention_mask, queries_dtype=queries.dtype)
            # We apply causal mask in attention instead of using is_causal=True.
            is_causal = False
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
                attn_mask=bias,
                is_causal=is_causal,
            )
            .transpose(1, 2)
            .contiguous()
        )


ATTN_ACTIVATIONS = {
    "relu": F.relu,
    "relu_squared": lambda x: torch.pow(F.relu(x), 2),
    # "gelu": F.gelu, # goes to NaN with bais so comment out for now
    "softplus": F.softplus,
    "identity": lambda x: x,
    "relu6": F.relu6,
    "sigmoid": F.sigmoid,
    "softmax": partial(F.softmax, dim=-1),
}

ATTN_SEQ_SCALARS = {
    "max": lambda x: x,
    # "seq": lambda x: torch.arange(x) + 1,  # comment out for now more involved
    "avg": lambda x: (x - 1) / 2 + 1,
    "none": lambda _: 1,
}


def custom_attn(
    queries,
    keys,
    values,
    attn_activation,
    attn_seq_scalar,
    alpha,
    is_causal=False,
    attention_mask=None,
) -> torch.Tensor:
    # naive reference implementation for relu-attention following: https://arxiv.org/pdf/2309.08586.pdf
    # code modifies: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if attention_mask is not None:
        raise NotImplementedError("attention_mask not yet implemented for custom_attn.")

    batch, q_seq_len, heads, embed_dim = queries.shape
    _, k_seq_len, _, _ = keys.shape

    attn_bias = torch.zeros(batch, heads, q_seq_len, k_seq_len, device=queries.device, dtype=queries.dtype)
    if is_causal and queries.shape[1] > 1:
        attn_bias = get_rectangular_causal_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)

    inner_scale = embed_dim**-0.5
    attn_weight = torch.einsum("bqhd,bkhd->bhqk", inner_scale * queries, keys)
    attn_weight += attn_bias

    # scaling by: 1/L^{-\alpha}
    outter_scale = ATTN_SEQ_SCALARS[attn_seq_scalar](k_seq_len) ** -alpha
    attn_weight = outter_scale * ATTN_ACTIVATIONS[attn_activation](attn_weight)

    return torch.einsum("bhqk,bkhd->bqhd", attn_weight, values)


def get_attn_func(attn_name):
    if attn_name == "auto":
        return xformers_attn if torch.cuda.is_available() else torch_attn
    elif attn_name == "xformers_attn":
        return xformers_attn
    elif attn_name == "xformers_attn_variable_length":
        # Upon changing the input sequence length, xformers attention changes
        # the stride dimension of the output tensor. This makes future calls to
        # .view() that collapses last two dimensions fail. One thus needs to
        # call .contiguous() on the output tensor. [#188]
        return lambda *args, **kwargs: xformers_attn(*args, **kwargs).contiguous()
    elif attn_name == "torch_attn":
        return torch_attn
    else:
        raise ValueError(f"Unsupported attn-name: {attn_name}")