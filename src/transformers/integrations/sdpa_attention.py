import torch 


def sdpa_attention_forward(config, query, key, value, mask, **_kwargs):
    key = repeat_kv(key, config.num_key_value_groups)
    value = repeat_kv(value, config.num_key_value_groups)

    causal_mask = mask
    if mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query.device.type == "cuda" and causal_mask is not None:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and query.shape[1] > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=config.attention_dropout if config.training else 0.0,
        is_causal=is_causal,
        scale=config.scaling,
    )
    return attn_output, None