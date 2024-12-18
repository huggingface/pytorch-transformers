# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import sys

import torch
from deepspeed import initialize

from transformers import AutoModel
from transformers.integrations.deepspeed import is_deepspeed_sp_enabled  # noqa
from transformers.modeling_flash_attention_utils import _flash_attention_forward


# Call transformer flash attention with and without deepspeed sp enabled and compare they match
def test_transformer_flash_attention(seq_len=2) -> None:
    model = AutoModel.from_pretrained("bert-base-uncased")
    batch_size = 2

    # Test with deepspeed sp
    sp_size = 2
    dp_size = 1
    ds_engine, _, _, _ = initialize(
        model=model,
        config_params={
            "train_batch_size": batch_size,
            "data_parallel_size": dp_size,
            "sequence_parallel_size": sp_size,
        },
    )

    assert is_deepspeed_sp_enabled()

    seq_len = seq_len
    hidden_dim = 16
    num_heads = 4
    head_dim = hidden_dim // num_heads
    # Create input tensors
    input_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim, device=ds_engine.device)
    input_tensor = input_tensor.half()
    attention_mask = None
    q, k, v = input_tensor, input_tensor, input_tensor

    output_tensor = _flash_attention_forward(q, k, v, attention_mask, query_length=seq_len, is_causal=False)
    assert output_tensor is not None
    assert output_tensor.shape == (batch_size, seq_len, num_heads, head_dim)

    # Now test without deepspeed sp
    sp_size = 1
    dp_size = 2
    ds_engine, _, _, _ = initialize(
        model=model,
        config_params={
            "train_batch_size": batch_size,
            "data_parallel_size": dp_size,
            "sequence_parallel_size": sp_size,
        },
    )
    assert not is_deepspeed_sp_enabled()

    output_tensor_no_sp = _flash_attention_forward(q, k, v, attention_mask, query_length=seq_len, is_causal=False)
    assert output_tensor_no_sp is not None
    assert output_tensor_no_sp.shape == (batch_size, seq_len, num_heads, head_dim)
    assert torch.allclose(output_tensor, output_tensor_no_sp)


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len = int((sys.argv[2]).split("=")[1])
    test_transformer_flash_attention(seq_len=seq_len)
