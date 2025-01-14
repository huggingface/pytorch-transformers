# This example demonstrates how to implement a "Zero-Delay QKV Compression" approach during inference 
# with a Hugging Face Transformer model. The idea is to modify the model's attention mechanism to 
# combine the query/key/value (Q/K/V) projections into a single linear operation. 
#
# Note:
# - This is a conceptual example. It shows how you might integrate a single QKV projection for 
#   runtime optimization, rather than running three separate projections. Actual performance improvements 
#   depend on hardware, model size, and other factors.
# - We use a GPT-2 model as a base example. Other models in the Transformers library share similar 
#   attention architectures but may differ slightly in naming or internal structure.
# - This code uses monkey-patching: after loading the model, we replace the original attention 
#   projection modules with a combined QKV module.
# - The zero-delay QKV compression technique essentially merges Q, K, V linear layers into a single 
#   linear layer that outputs concatenated Q, K, and V. This can potentially reduce overhead and 
#   improve inference speed.
#
# Prerequisites:
# - pip install transformers
# - Ensure you have a GPU available if you're testing speed improvements.

from transformers import GPT2Model, GPT2Config
import torch
import torch.nn as nn

# -----------------------------
# Define a combined QKV projection module
# -----------------------------
class CombinedQKVProjection(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Store essential parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Instead of separate W_q, W_k, W_v, we use a single W_qkv.
        # The output dimension is 3 * embed_dim since we need Q, K, and V concatenated.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, embed_dim)
        
        # Single linear projection for Q, K, V
        qkv_out = self.qkv(hidden_states)  # shape: (batch_size, seq_len, 3 * embed_dim)
        
        # Split the projections into Q, K, V
        # The shape after split will be:
        # Q: (batch_size, seq_len, embed_dim)
        # K: (batch_size, seq_len, embed_dim)
        # V: (batch_size, seq_len, embed_dim)
        q, k, v = qkv_out.split(self.embed_dim, dim=-1)
        
        # Now we reshape them to (batch_size, num_heads, seq_len, head_dim) as required by multi-head attention
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (bs, heads, seq, head_dim)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (bs, heads, seq, head_dim)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (bs, heads, seq, head_dim)
        
        return q, k, v


# -----------------------------
# Function to replace the original Q/K/V with combined QKV in GPT-2
# -----------------------------
def replace_attn_qkv_with_combined(model):
    # GPT-2 uses a module called `CausalSelfAttention` inside each Transformer block
    # This module typically has `c_attn` (a single linear layer) that already produces q, k, v
    # as a single projection. GPT-2 in HuggingFace Transformers actually already uses a combined QKV projection,
    # but for demonstration, let's assume a scenario where Q, K, and V are separate and we want to merge them.
    #
    # For GPT-2, the c_attn layer already does QKV in one step:
    # self.c_attn = Conv1D(3 * n_state, n_state)
    #
    # In many other models (like BERT), you'd see something like:
    # self.query = nn.Linear(embed_dim, embed_dim)
    # self.key = nn.Linear(embed_dim, embed_dim)
    # self.value = nn.Linear(embed_dim, embed_dim)
    #
    # For demonstration, let's pretend we must replace GPT-2's c_attn with our CombinedQKVProjection.
    #
    # Steps:
    # 1. Identify each attention block in the model.
    # 2. Extract the necessary parameters (embed_dim, num_heads).
    # 3. Replace the c_attn layer with our CombinedQKVProjection.

    # GPT-2 config details: 
    # n_embd: dimension of embeddings
    # n_head: number of attention heads
    embed_dim = model.config.n_embd
    num_heads = model.config.n_head

    # Iterate over all transformer blocks
    for block in model.h:
        # block.attn is a CausalSelfAttention module
        # Save original weights and biases before replacing
        old_weight = block.attn.c_attn.weight.data
        old_bias = block.attn.c_attn.bias.data

        # Create our new combined QKV projection module
        new_qkv = CombinedQKVProjection(embed_dim, num_heads)

        # Load the old weights into the new module. GPT-2 c_attn layer has shape [3*embed_dim, embed_dim]
        # which matches our CombinedQKVProjection qkv layer.
        new_qkv.qkv.weight.data = old_weight
        new_qkv.qkv.bias.data = old_bias

        # Replace the old c_attn module with the new combined qkv module
        block.attn.c_attn = new_qkv

    return model

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    # Create a GPT-2 model
    config = GPT2Config()
    model = GPT2Model(config)

    # Before optimization: model uses its original QKV projection (already combined in GPT-2, 
    # but let's assume we want to enforce our own CombinedQKVProjection).
    # After optimization: we explicitly ensure Q,K,V are projected via a single combined linear operation.
    model = replace_attn_qkv_with_combined(model)

    # Run a forward pass to confirm the model still works:
    input_ids = torch.randint(0, config.vocab_size, (1, 10))  # a random input sequence of length 10
    outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    print("Output shape:", last_hidden_state.shape)
    # You should see something like: torch.Size([1, 10, 768]) for a base GPT-2 config.

    # Note: This code is illustrative. Actual zero-delay QKV compression and runtime optimization may
    # involve additional steps such as quantization, specialized kernels, or integration with hardware-specific 
    # libraries like DeepSpeed or TensorRT. However, this demonstrates how to modify a HuggingFace model 
    # to use a single QKV projection for potential speed-ups.
