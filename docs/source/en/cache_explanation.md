<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Caching

Imagine you’re having a conversation with someone, and instead of remembering what they previously said, they have to start from scratch every time you respond. This would be slow and inefficient, right?

You can extend this analogy to transformer models. Autoregressive model generation can be slow because it makes a prediction one token at a time. Each new prediction is dependent on all the previous context.

To predict the 1000th token, the model requires information from the previous 999 tokens. The information is represented as matrix multiplications across the token representations.

To predict the 1001th token, you need the same information from the previous 999 tokens in addition to any information from the 1000th token. This is a lot of matrix multiplications a model has to compute over and over for each token!

A key-value (KV) cache eliminates this inefficiency by storing kv pairs derived from the self-attention layers of previously processed tokens. The stored kv pairs are retrieved from the cache and reused for subsequent tokens, avoiding the need to recompute.

> [!WARNING]
> Caching should only be used for **inference**. It may cause unexpected errors if it's enabled during training.

## Cache class

When you use Transformers' [`Cache`] class, the attention module performs several critical steps to integrate past and present information.

1. The attention module concatenates current kv pairs with past kv pairs stored in the cache. This creates attentions weights with the shape `(new_tokens_length, past_kv_length + new_tokens_length)`. The current and past kv pairs are essentially combined to compute the attention scores, ensuring a model is aware of previous context and the current input.

2. When the `forward` method is called iteratively, it's crucial that the attention mask shape matches the combined length of the past and current kv pairs. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is typically handled internally in [`~GenerationMixin.generate`], but if you want to implement your own generation loop with [`Cache`], keep this in mind! The attention mask should hold the past and current token values.

3. It is also important to be aware of the `cache_position`. This is important if you want to reuse a prefilled [`Cache`] with the `forward` method because you have to pass a valid `cache_position` value. This indicates the input positions in a sequence. `cache_position` is unaffected by padding, and it always adds one more position for each token. For example, if a kv cache contains 10 tokens - regardless of pad tokens - the cache position for the next token should be `torch.tensor([10]).

The example below demonstrates how to create a generation loop with [`DynamicCache`]. As discussed, the attention mask is a concatenation of past and current token values and `1` is added to the cache position for the next token.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache()
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda:0")

generated_ids = inputs.input_ids
cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device="cuda:0")
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
    # Greedily sample one next token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
    # and expanding attn mask for the new token, as explained above
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    cache_position = cache_position[-1:] + 1 # add one more position for the next token

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
```
