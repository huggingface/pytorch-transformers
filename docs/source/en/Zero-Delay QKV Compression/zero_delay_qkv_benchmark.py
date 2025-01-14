"""
This script provides a full example demonstrating how to implement and benchmark 
a "Zero-Delay QKV Compression" technique with HuggingFace Transformers. We will:

1. Use a BERT model as a reference since it typically uses separate Q/K/V projections.
2. Replace the separate query, key, and value linear layers in each attention head 
   with a single combined QKV projection layer.
3. Benchmark the inference speed before and after this modification.

Note: 
- Actual performance gains depend heavily on your hardware, model size, and 
  other factors (like whether you are using GPU, whether you’ve compiled custom kernels, etc.).
- This example is conceptual and shows how to modify the attention modules. 
  Results may vary and might not show large improvements out-of-the-box.

Prerequisites:
- pip install transformers
- pip install torch
- Ensure you have a GPU if you want to benchmark GPU performance.

Run:
python3 zero_delay_qkv_benchmark.py
"""

import time
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# ============================================================
# Define a combined QKV projection module
# ============================================================
class CombinedQKVProjection(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # One linear layer outputs Q, K, and V concatenated: dimension is 3 * embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, embed_dim)
        qkv_out = self.qkv(hidden_states)  # (batch_size, seq_len, 3 * embed_dim)
        
        # Split into Q, K, V
        q, k, v = qkv_out.split(self.embed_dim, dim=-1)
        
        # Reshape for multi-head attention: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        return q, k, v


# ============================================================
# Function to replace Q/K/V in a BERT model with the combined QKV
# ============================================================
def replace_attn_qkv_with_combined(model):
    # In BERT, each layer has a BertAttention module, which contains:
    # - self.self.query
    # - self.self.key
    # - self.self.value
    #
    # Each is a linear layer. We will replace them with a single CombinedQKVProjection.
    
    embed_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    for layer in model.encoder.layer:
        # Extract original weights and biases
        W_q = layer.attention.self.query.weight.data
        b_q = layer.attention.self.query.bias.data
        W_k = layer.attention.self.key.weight.data
        b_k = layer.attention.self.key.bias.data
        W_v = layer.attention.self.value.weight.data
        b_v = layer.attention.self.value.bias.data

        # Create the combined QKV layer
        combined_qkv = CombinedQKVProjection(embed_dim, num_heads)

        # The combined weight should be [3*embed_dim, embed_dim]
        # We need to stack query, key, value weights along the output dimension
        combined_qkv.qkv.weight.data = torch.cat([W_q, W_k, W_v], dim=0)
        combined_qkv.qkv.bias.data = torch.cat([b_q, b_k, b_v], dim=0)

        # Now we need to replace the forward logic. The BERT's SelfAttention calls:
        # query_layer = self.transpose_for_scores(self.query(hidden_states))
        # key_layer = self.transpose_for_scores(self.key(hidden_states))
        # value_layer = self.transpose_for_scores(self.value(hidden_states))
        #
        # We must ensure the attention computations now use combined_qkv.
        # One approach: monkey-patch the "forward" method of attention.self to use our combined_qkv.

        # Keep a reference to the original forward method and parameters if needed
        original_self_attn = layer.attention.self

        # Define a custom forward method that uses the combined projection
        def custom_forward(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                           encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            # If encoder_hidden_states are present, we attend them instead of ourselves.
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention:
                # For cross-attention, keys and values come from encoder_hidden_states
                q, _, _ = combined_qkv(hidden_states)
                # We still use original separate projections for keys and values in cross-attention or 
                # you can also add logic to handle cross-attention QKV in a combined manner if required.
                # For simplicity, let's just do combined projection on hidden_states (the query) and 
                # keep original logic for keys/values from encoder states:
                k = original_self_attn.transpose_for_scores(original_self_attn.key(encoder_hidden_states))
                v = original_self_attn.transpose_for_scores(original_self_attn.value(encoder_hidden_states))
            else:
                q, k, v = combined_qkv(hidden_states)

            # If a previous key/value states are provided (past_key_value), concatenate them at the seq dimension
            if past_key_value is not None:
                # reuse k and v
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

            # Update present_key_value
            present_key_value = (k, v) if original_self_attn.is_decoder else None

            # Calculate attention scores
            attn_scores = torch.matmul(q, k.transpose(-1, -2))
            attn_scores = attn_scores / (original_self_attn.sqrt_att_head_size)

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            # Attention probabilities
            attn_probs = nn.Softmax(dim=-1)(attn_scores)

            if head_mask is not None:
                attn_probs = attn_probs * head_mask

            # Apply attention
            context = torch.matmul(attn_probs, v)

            # Transpose back
            context = context.transpose(1, 2).contiguous().view(context.size(0), context.size(2), embed_dim)

            # Output
            outputs = (context, attn_probs if output_attentions else None)
            if original_self_attn.is_decoder:
                outputs = outputs + (present_key_value,)

            return outputs

        # Replace the self-attention's forward method and remove old query/key/value modules
        original_self_attn.forward = custom_forward
        del original_self_attn.query
        del original_self_attn.key
        del original_self_attn.value

    return model

# ============================================================
# Benchmarking function
# ============================================================
def benchmark_model(model, input_ids, attention_mask, warmup=5, trials=50, device='cuda'):
    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Warmup runs
    for _ in range(warmup):
        _ = model(input_ids, attention_mask=attention_mask)

    # Timed runs
    start = time.time()
    torch.cuda.synchronize(device)  # Make sure GPU is ready
    for _ in range(trials):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize(device)  # Wait for all runs to finish
    end = time.time()

    avg_time = (end - start) / trials
    return avg_time

# ============================================================
# Example usage and benchmarking
# ============================================================
if __name__ == "__main__":
    # Create a BERT model and some dummy input
    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        vocab_size=30522,
    )
    model = BertModel(config)

    # Dummy input: batch size = 8, seq_length = 64
    batch_size = 8
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # Benchmark original model
    original_time = benchmark_model(model, input_ids, attention_mask, device=device)
    print(f"Original model average inference time per batch: {original_time*1000:.2f} ms")

    # Replace Q/K/V with combined QKV
    model = replace_attn_qkv_with_combined(model)

    # Benchmark modified model
    combined_time = benchmark_model(model, input_ids, attention_mask, device=device)
    print(f"Combined QKV model average inference time per batch: {combined_time*1000:.2f} ms")

    # Print speedup
    if combined_time < original_time:
        speedup = original_time / combined_time
        print(f"Speedup: {speedup:.2f}x faster")
    else:
        print("No speedup observed (combined is slower or roughly the same)")
