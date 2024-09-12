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

# KV cache strategies

The key-value (KV) vectors are used to calculate attention scores, and for autoregressive models, the KV scores are calculated *every* time because the model predicts one token at a time. Each prediction depends on the previous tokens, which means the model performs the same computations each time.

A KV *cache* stores these calculations so they can be reused without recomputing them. Efficient caching is crucial for optimizing model performance because it reduces computation time and improves response rates. For a more in-depth explanation about how a cache works, refer to [Caching](./cache_explanation.md).

Transformers offers several [`Cache`] classes that implement different caching mechanisms. Some of these [`Cache`] classes are optimized to save memory while others are designed to maximize generation speed. Refer to the table below to compare cache types and use it to help you select the best cache for your use case.

| Cache Type             | Memory Efficient  | Supports torch.compile() | Initialization Recommended | Latency | Long Context Generation |
|------------------------|------------------|--------------------------|----------------------------|---------|-------------------------|
| Dynamic Cache          | No               | No                       | No                         | Mid     | No                      |
| Static Cache           | No               | Yes                      | Yes                        | High    | No                      |
| Offloaded Cache         | Yes              | No                       | No                         | Low     | Yes                     |
| Offloaded Static Cache  | No               | Yes                      | Yes                        | High    | Yes                     |
| Quantized Cache        | Yes              | No                       | No                         | Low     | Yes                     |
| Sliding Window Cache   | No               | Yes                      | Yes                        | High    | No                      |
| Sink Cache             | Yes              | No                       | Yes                        | Mid     | Yes                     |

This guide introduces you to the different [`Cache`] classes and shows you how to use them for generation.

## Default cache

The [`DynamicCache`] is the default cache class for most models. It allows the cache size to grow dynamically in order to store an increasing number of keys and values as generation progresses.

Disable the cache by configuring `use_cache=False` in [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)
```

Cache classes can also be initialized first before calling and passing it to the models [past_key_values](https://hf.co/docs/transformers/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput.past_key_values) parameter. This cache initialization strategy is only recommended for some cache types.

In most other cases, it's easier to define the cache strategy in the [cache_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) parameter.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

past_key_values = DynamicCache()
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, past_key_values=past_key_values)
```

## Memory efficient caches

The KV cache can occupy a significant portion of memory and become a [bottleneck](https://hf.co/blog/llama31#inference-memory-requirements) for long-context generation. Memory efficient caches focus on trading off speed for reduced memory usage. This is especially important for large language models (LLMs) and if your hardware is memory constrained.

### Offloaded cache

The [`OffloadedCache`] saves GPU memory by moving the KV cache for most model layers to the CPU. Only the current layer cache is maintained on the GPU during a models `forward` iteration over the layers. [`OffloadedCache`] asynchronously prefetches the next layer cache and sends the previous layer cache back to the CPU.

This cache strategy always generates the same result as [`DynamicCache`] and works as a drop-in replacement or fallback. You may want to use [`OffloadedCache`] if you have a GPU and you're getting out-of-memory (OOM) errors.

> [!WARNING]
> You may notice a small degradation in generation throughput compared to [`DynamicCache`] depending on your model and generation choices (context size, number of generated tokens, number of beams, etc.).

Enable [`OffloadedCache`] by configuring `cache_implementation="offloaded"` in either [`GenerationConfig`] or [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("Fun fact: The shortest", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=23, cache_implementation="offloaded")
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.
```

The example below shows how you can fallback on [`OffloadedCache`] if you run out of memory.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def resilient_generate(model, *args, **kwargs):
    oom = False
    try:
        return model.generate(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        print("retrying with cache_implementation='offloaded'")
        oom = True
    if oom:
        torch.cuda.empty_cache()
        kwargs["cache_implementation"] = "offloaded"
        return model.generate(*args, **kwargs)

ckpt = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda:0")
prompt = ["okay "*1000 + "Fun fact: The most"]
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
beams = { "num_beams": 40, "num_beam_groups": 40, "num_return_sequences": 40, "diversity_penalty": 1.0, "max_new_tokens": 23, "early_stopping": True, }
out = resilient_generate(model, **inputs, **beams)
responses = tokenizer.batch_decode(out[:,-28:], skip_special_tokens=True)
```

### Quantized cache

The [`QuantizedCache`] reduces memory requirements by quantizing the KV values to a lower precision. [`QuantizedCache`] currently supports two quantization backends.

- [`HQQQuantizedCache`] supports int2, int4, and int8 datatypes.
- [`QuantoQuantizedCache`] supports int2 and int4 datatypes. This is the default quantization backend.

> [!WARNING]
> Quantizing the cache can harm latency if the context length is short and there is enough GPU memory available for generation without enabling cache quantization. Try to find a balance between memory efficiency and latency.

Enable [`QuantizedCache`] by configuring `cache_implementation="quantized"` in [`GenerationConfig`], and indicate the quantization backend in [`QuantizedCacheConfig`]. Any additional quantization related parameters should also be passed either as a dict or an instance of [`QuantizedCacheConfig`]. You should use the default values for these additional parameters unless you're running out-of-memory. In that case, consider decreasing the residual length.

<hfoptions id="quantized-cache">
<hfoption id="HQQQuantizedCache">

For [`HQQQuantizedCache`], we recommend setting the `axis-key` and `axis-value` parameters to `1`.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"axis-key": 1, "axis-value": 1, "backend": "hqq"})
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel
```

</hfoption>
<hfoption id="Quanto">

For [`QuantoQuantizedCache`], we recommend setting the `axis-key` and `axis-value` parameters to `0`.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, QuantizedCacheConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4, "axis-key": 0, "axis-value": 0, "backend": "quanto"})
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel
```

</hfoption>
</hfoptions>

### Sink cache

[`SinkCache`] is capable of generating very long sequences ("infinite length" according to the paper) by only retaining a few initial tokens from the sequence. These are called the *sink tokens* because they account for a significant portion of the attention scores during generation. Subsequent tokens are discarded on a sliding windowed basis, and only the latest `window_size` tokens are kept. This means most of the previous knowledge is discarded.

The sink tokens allow a model to maintain stable performance even when it's dealing with very long text sequences.

Enable [`SinkCache`] by initializing it first with the [window_length](https://hf.co/docs/transformers/main/en/internal/generation_utils#transformers.SinkCache.window_length) and [num_sink_tokens](https://hf.co/docs/transformers/main/en/internal/generation_utils#transformers.SinkCache.num_sink_tokens) parameters before passing it to [past_key_values](https://hf.co/docs/transformers/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput.past_key_values) in [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("This is a long story about unicorns, fairies and magic.", return_tensors="pt").to(model.device)

past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
out = model.generate(**inputs, do_sample=False, max_new_tokens=30, past_key_values=past_key_values)
tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"This is a long story about unicorns, fairies and magic. It is a fantasy world where unicorns and fairies live together in harmony. The story follows a young girl named Lily"
```

## Speed optimized caches

The default [`DynamicCache`] prevents you from taking advantage of just-in-time (JIT) optimizations because the cache size isn't fixed. JIT optimizations enable you to maximize latency at the expense of memory usage. All of the following cache types are compatible with JIT optimizations like [torch.compile](./llm_optims#static-kv-cache-and-torchcompile) to accelerate generation.

### Static cache

A [`StaticCache`] pre-allocates a specific maximum cache size for the kv pairs. You can generate up to the maximum cache size without needing to modify it.

Enable [`StaticCache`] by configuring `cache_implementation="static"` in [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="static")
tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of"
```

### Offloaded static cache

The [`OffloadedStaticCache`] is very similar to the [OffloadedCache](#offloaded-cache) except the cache size is set to a maximum cache size. Otherwise, [`OffladedStaticCache`] only keeps the current layer cache on the GPU and the rest are moved to the CPU.

Enable [`OffloadedStaticCache`] by configuring `cache_implementation="offloaded_static"` in [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="offloaded_static")
tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of"
```
Cache offloading requires a CUDA GPU.

### Sliding window cache

[`SlidingWindowCache`] implements a sliding window over the previos kv pairs, and only keeps the last `sliding_window` tokens. This cache type is designed to only work with models that support *sliding window attention*, such as [Mistral](./model_doc/mistral). Older kv states are discarded and replaced by new kv states.

Enable [`SlidingWindowCache`] by configuring `cache_implementation="sliding_window"` in [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("Yesterday I was on a rock concert and.", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=30, cache_implementation="sliding_window")
tokenizer.batch_decode(out, skip_special_tokens=True)[0]
```

## Model caches

Some model types, like encoder-decoder models or [Gemma2](./model_doc/gemma2) and [Mamba](./model_doc/mamba), have dedicated cache classes.

### Encoder-decoder cache

[`EncoderDecoderCache`] is designed for encoder-decoder models. It manages both the self-attention and cross-attention caches to ensure storage and retrieval of previous kv pairs. It is possible to individually set a different cache type for the encoder and decoder.

This cache type doesn't require any setup. It can be used when calling [`~GenerationMixin.generate`] or a models `forward` method.

> [!TIP]
> The [`EncoderDecoderCache`] currently only supports [Whisper](./model_doc/whisper).

### Model-specific caches

Some models have a unique way of storing past kv pairs or states that is not compatible with any other cache classes.

[Gemma2](./model_doc/gemma2) requires [`HybridCache`], which uses a combination of [`SlidingWindowCache`] for sliding window attention and [`StaticCache`] for global attention under the hood.

[Mamba](./model_doc/mamba) requires [`MambaCache`] because the model doesn't have an attention mechanism or kv states.

## Iterative generation

A cache can also work in iterative generation settings where there is back-and-forth interaction with a model (chatbots). Like regular generation, iterative generation with a cache allows a model to efficiently handle ongoing conversations without recomputing the entire context at each step.

For iterative generation with a cache, start by initializing an empty cache class and then you can feed in your new prompts. Keep track of dialogue history with a [chat template](./chat_templating).

If you're using [`SinkCache`], the inputs need to be truncated to the maximum length because [`SinkCache`] can generate text that exceeds its maximum window size. However, the first input shouldn't exceed the maximum cache length.

The example below demonstrates how to use a cache for iterative generation.

```py
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

user_prompts = ["Hello, what's your name?", "Btw, yesterday I was on a rock concert."]

past_key_values = DynamicCache()
max_cache_length = past_key_values.get_max_length()

messages = []
for prompt in user_prompts:
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    if isinstance(past_key_values, SinkCache):
        inputs = {k: v[:, -max_cache_length:] for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=past_key_values)
    completion = tokenizer.decode(outputs[0, input_length: ], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})
```

## Prefill a cache

In some situations, you may want to fill a [`Cache`] with kv pairs for a certain prefix prompt and reuse it to generate different sequences.

The example below initializes a [`StaticCache`], and then caches an initial prompt. Now you can generate several sequences from the prefilled prompt.

```py
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Init StaticCache with big enough max-length (1024 tokens for the below example) 
# You can also init a DynamicCache, if that suits you better
prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=1024, device="cuda", dtype=torch.bfloat16)

INITIAL_PROMPT = "You are a helpful assistant. "
inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to("cuda")
# This is the common prompt cached, we need to run forward without grad to be able to copy
with torch.no_grad():
     prompt_cache = model(**inputs_initial_prompt, past_key_values = prompt_cache).past_key_values

prompts = ["Help me to write a blogpost about travelling.", "What is the capital of France?"]
responses = []
for prompt in prompts:
    new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to("cuda")
    past_key_values = copy.deepcopy(prompt_cache)
    outputs = model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=20) 
    response = tokenizer.batch_decode(outputs)[0]
    responses.append(response)

print(responses)
```
