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

# Best Practices for Generation with Cache

Efficient caching is crucial for optimizing the performance of models in various generative tasks,
including text generation, translation, summarization and other transformer-based applications.
Effective caching helps reduce computation time and improve response rates, especially in real-time or resource-intensive applications.

Transformers support various caching methods, leveraging "Cache" classes to abstract and manage the caching logic.
This document outlines best practices for using these classes to maximize performance and efficiency.
Check out all the available `Cache` classes in the [API documentation](./internal/generation_utils).

## What is Cache and why we should care?

Imagine you’re having a conversation with someone, and instead of remembering what was said previously, you have to start from scratch every time you respond. This would be slow and inefficient, right? In the world of Transformer models, a similar concept applies, and that's where Caching keys and values come into play. From now on, I'll refer to the concept as KV Cache.

KV cache is needed to optimize the generation in autoregressive models, where the model predicts text token by token. This process can be slow since the model can generate only one token at a time, and each new prediction is dependent on the previous context. That means, to predict token number 1000 in the generation, you need information from the previous 999 tokens, which comes in the form of some matrix multiplications across the representations of those tokens. But to predict token number 1001, you also need the same information from the first 999 tokens, plus additional information from token number 1000. That is where key-value cache is used to optimize the sequential generation process by storing previous calculations to reuse in subsequent tokens, so they don't need to be computed again.

More concretely, key-value cache acts as a memory bank for these generative models, where the model stores key-value pairs derived from self-attention layers for previously processed tokens. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previous tokens from the cache. Note that caching can be used only in inference and should be disabled when training, otherwise it might cause unexpected errors.

<details>
  <summary><em>For the Curious Minds Who Like to Dive Deep</em></summary>

  ### Under the Hood: How Cache Object Works in Attention Mechanism

  When utilizing a cache object in the input, the Attention module performs several critical steps to integrate past and present information seamlessly.

  The Attention module concatenates the current key-values with the past key-values stored in the cache. This results in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`. Essentially, the past and current key-values are combined to compute attention scores, ensuring that the model considers both previous context and new input. The concatenated key-values are used to compute the attention scores resulting in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`.

  Therefore, when iteratively calling `forward()` instead of the `generate()` method, it’s crucial to ensure that the attention mask shape matches the combined length of past and current key-values. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is usually handled internally when you call `generate()` method. If you want to implement your own generation loop with Cache classes, take this into consideration and prepare the attention mask to hold values to current and past tokens.

  <Tip warning={true}>

  One important concept you need to know when writing your own generation loop, is `cache_position`. In case you want to reuse an already filled Cache object by calling `forward()`, you have to pass in a valid `cache_position` which will indicate the positions of inputs in the sequence. Note that `cache_position` is not affected by padding, and always adds one more position for each token. For example, if key/value cache contains 10 tokens (no matter how many of it is a pad token), the cache position for the next token should be `torch.tensor([10])`.

  </Tip>


  See an example below for how to implement your own generation loop.

  ```python
  >>> import torch
  >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

  >>> model_id = "meta-llama/Llama-2-7b-chat-hf"
  >>> model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
  >>> tokenizer = AutoTokenizer.from_pretrained(model_id)

  >>> past_key_values = DynamicCache()
  >>> messages = [{"role": "user", "content": "Hello, what's your name."}]
  >>> inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda:0")

  >>> generated_ids = inputs.input_ids
  >>> cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device="cuda:0")
  >>> max_new_tokens = 10

  >>> for _ in range(max_new_tokens):
  ...     outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
  ...     # Greedily sample one next token
  ...     next_token_ids = outputs.logits[:, -1:].argmax(-1)
  ...     generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
  ...
  ...     # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
  ...     # and expanding attn mask for the new token, as explained above
  ...     attention_mask = inputs["attention_mask"]
  ...     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
  ...     inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
  ...     cache_position = cache_position[-1:] + 1 # add one more position for the next token

  >>> print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
  "[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
  ```

</details>



## Generate with Cache

In 🤗 Transformers, we support various Cache types to optimize the performance across different models and tasks. By default, all models generate with caching,
with the [`~DynamicCache`] class being the default cache for most models. It allows us to dynamically grow cache size, by saving more and more keys and values as we generate. If for some reason you don't want to use caches, you can pass `use_cache=False` into the `generate()` method.

Refer to the table below to see the difference between cache types and choose the one that suits best for your use-case. Models for which initialization is recommended should be initialized before calling the model and passed to model as a kwarg. In all other cases you can simply define desired `cache_implementation` and we take care of the rest for you.

| Cache Type             | Memory Efficient | Supports torch.compile() | Initialization Recommended | Latency | Long Context Generation |
|------------------------|------------------|--------------------------|----------------------------|---------|-------------------------|
| Dynamic Cache          | No               | No                       | No                         | Mid     | No                      |
| Static Cache           | No               | Yes                      | Yes                        | High    | No                      |
| Offloaded Cache        | Yes              | No                       | No                         | Low     | Yes                     |
| Offloaded Static Cache | No               | Yes                      | Yes                        | High    | Yes                     |
| Quantized Cache        | Yes              | No                       | No                         | Low     | Yes                     |
| Sliding Window Cache   | No               | Yes                      | Yes                        | High    | No                      |
| Sink Cache             | Yes              | No                       | Yes                        | Mid     | Yes                     |


These cache classes can be set with a `cache_implementation` argument when generating. To learn about the available options for the cache_implementation flag, please refer to the [API Documentation](./main_classes/text_generation#transformers.GenerationConfig). Now, let's explore each cache type in detail and see how to use them. Note that the below examples are for decoder-only Tranformer-based models. We also support ["Model-Specific Cache"] classes for models such as Mamba or Jamba, keep reading for more details.

### Quantized Cache

The key and value cache can occupy a large portion of memory, becoming a [bottleneck for long-context generation](https://huggingface.co/blog/llama31#inference-memory-requirements), especially for Large Language Models.
Quantizing the cache when using `generate()` can significantly reduce memory requirements at the cost of speed.

KV Cache quantization in `transformers` is largely inspired by the paper ["KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"](https://arxiv.org/abs/2402.02750) and currently supports [`~QuantoQuantizedCache`] and [`~HQQQuantizedCache`] classes. For more information on the inner workings see the paper.

To enable quantization of the key-value cache, one needs to indicate `cache_implementation="quantized"` in the `generation_config`.
Quantization related arguments should be passed to the `generation_config` either as a `dict` or an instance of a [`~QuantizedCacheConfig`] class.
One has to indicate which quantization backend to use in the [`~QuantizedCacheConfig`], the default is `quanto`.

It is recommended to set `axis-key/axis-value` parameters in the cache config to `0` if you're using the `quanto` backend and to `1` if you're using the `HQQ` backend. For other config values, please use the defaults unless you're running out of memory. In that case, you may consider decreasing the residual length.

<Tip warning={true}>

Cache quantization can be detrimental in terms of latency if the context length is short and there is enough GPU VRAM available to run without cache quantization. It is recommended to seek balance between memory efficiency and latency.
</Tip>


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4, "backend": "quanto"})
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. I like to listen to it when I'm feeling
```

### Offloaded Cache

Similarly to KV cache quantization, [`~OffloadedCache`] strategy aims to reduce GPU VRAM usage.
It does so by moving the KV cache for most layers to the CPU.
As the model's `forward()` method iterates over the layers, this strategy maintains the current layer cache on the GPU.
At the same time it asynchronously prefetches the next layer cache as well as sending the previous layer cache back to the CPU.
Unlike KV cache quantization, this strategy always produces the same result as the default KV cache implementation.
Thus, it can serve as a drop-in replacement or a fallback for it.

Depending on your model and the characteristics of your generation task (size of context, number of generated tokens, number of beams, etc.)
you may notice a small degradation in generation throughput compared to the default KV cache implementation.

To enable KV cache offloading, pass `cache_implementation="offloaded"` in the `generation_config` or directly to the `generate()` call.
Use `cache_implementation="offloaded_static"` for an offloaded static cache (see also [Offloaded Static Cache](#offloaded-static-cache) below).

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> ckpt = "microsoft/Phi-3-mini-4k-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(ckpt)
>>> model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("Fun fact: The shortest", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=23, cache_implementation="offloaded")
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=23)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.
```

<Tip warning={true}>

Cache offloading requires a CUDA GPU and can be slower than dynamic KV cache. Use it if you are getting CUDA out of memory errors.

</Tip>

The example below shows how KV cache offloading can be used as a fallback strategy.
```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> def resilient_generate(model, *args, **kwargs):
...     oom = False
...     try:
...         return model.generate(*args, **kwargs)
...     except torch.cuda.OutOfMemoryError as e:
...         print(e)
...         print("retrying with cache_implementation='offloaded'")
...         oom = True
...     if oom:
...         torch.cuda.empty_cache()
...         kwargs["cache_implementation"] = "offloaded"
...         return model.generate(*args, **kwargs)
...
...
>>> ckpt = "microsoft/Phi-3-mini-4k-instruct"
>>> tokenizer = AutoTokenizer.from_pretrained(ckpt)
>>> model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda:0")
>>> prompt = ["okay "*1000 + "Fun fact: The most"]
>>> inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
>>> beams = { "num_beams": 40, "num_beam_groups": 40, "num_return_sequences": 40, "diversity_penalty": 1.0, "max_new_tokens": 23, "early_stopping": True, }
>>> out = resilient_generate(model, **inputs, **beams)
>>> responses = tokenizer.batch_decode(out[:,-28:], skip_special_tokens=True)
```

On a GPU with 50 GB of RAM, running this code will print
```
CUDA out of memory. Tried to allocate 4.83 GiB. GPU
retrying with cache_implementation='offloaded'
```
before successfully generating 40 beams.


### Static Cache

Since the "DynamicCache" dynamically grows with each generation step, it prevents you from taking advantage of JIT optimizations. The [`~StaticCache`] pre-allocates
a specific maximum size for the keys and values, allowing you to generate up to the maximum length without having to modify cache size. Check the below usage example.

For more examples with Static Cache and JIT compilation, take a look at [StaticCache & torchcompile](./llm_optims#static-kv-cache-and-torchcompile)

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # simply pass the cache implementation="static"
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="static")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of"
```


## Offloaded Static Cache

Like [`~OffloadedCache`] exists for offloading a "DynamicCache", there is also an offloaded static cache. It fully supports
JIT optimizations. Just pass `cache_implementation="offloaded_static"` in the `generation_config` or directly to the `generate()` call.
This will use the [`~OffloadedStaticCache`] implementation instead.

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # simply pass the cache implementation="static"
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="offloaded_static")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of"
```
Cache offloading requires a CUDA GPU.


### Sliding Window Cache

As the name suggests, this cache type implements a sliding window over previous keys and values, retaining only the last `sliding_window` tokens. It should be used with models like Mistral that support sliding window attention. Additionally, similar to Static Cache, this one is JIT-friendly and can be used with the same compile tecniques as Static Cache.

Note that you can use this cache only for models that support sliding window, e.g. Mistral models.


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("Yesterday I was on a rock concert and.", return_tensors="pt").to(model.device)

>>> # can be used by passing in cache implementation
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=30, cache_implementation="sliding_window")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Yesterday I was on a rock concert and. I was so excited to see my favorite band. I was so excited that I was jumping up and down and screaming. I was so excited that I"
```

### Sink Cache

Sink Cache was introduced in ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453). It allows you to generate long sequences of text ("infinite length" according to the paper) without any fine-tuning. That is achieved by smart handling of previous keys and values, specifically it retains a few initial tokens from the sequence, called "sink tokens". This is based on the observation that these initial tokens attract a significant portion of attention scores during the generation process. Tokens that come after "sink tokens" are discarded on a sliding windowed basis, keeping only the latest `window_size` tokens. By keeping these initial tokens as "attention sinks," the model maintains stable performance even when dealing with very long texts, thus discarding most of the previous knowledge.

Unlike other cache classes, this one can't be used directly by indicating a `cache_implementation`. You have to initialize the Cache before calling on `generate()` as follows.

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("This is a long story about unicorns, fairies and magic.", return_tensors="pt").to(model.device)

>>> # get our cache, specify number of sink tokens and window size
>>> # Note that window size already includes sink tokens, so has to be larger
>>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=30, past_key_values=past_key_values)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"This is a long story about unicorns, fairies and magic. It is a fantasy world where unicorns and fairies live together in harmony. The story follows a young girl named Lily"
```

### Encoder-Decoder Cache

The [`~EncoderDecoderCache`] is a wrapper designed to handle the caching needs of encoder-decoder models. This cache type is specifically built to manage both self-attention and cross-attention caches, ensuring storage and retrieval of past key/values required for these complex models. Cool thing about Encoder-Decoder Cache is that you can set different cache types for the encoder and for the decoder, depending on your use case. Currently this cache is only supported in [Whisper](./model_doc/whisper) models but we will be adding more models soon.

In terms of usage, there is nothing special to be done and calling `generate()` or `forward()` will handle everything for you.


### Model-specific Cache Classes

Some models require storing previous keys, values, or states in a specific way, and the above cache classes cannot be used. For such cases, we have several specialized cache classes that are designed for specific models. These models only accept their own dedicated cache classes and do not support using any other cache types. Some examples include [`~HybridCache`] for [Gemma2](./model_doc/gemma2) series models or [`~MambaCache`] for [Mamba](./model_doc/mamba) architecture models.


## Iterative Generation with Cache

We have seen how to use each of the cache types when generating. What if you want to use cache in iterative generation setting, for example in applications like chatbots, where interactions involve multiple turns and continuous back-and-forth exchanges. Iterative generation with cache allows these systems to handle ongoing conversations effectively without reprocessing the entire context at each step. But there are some tips that you should know before you start implementing:

The general format when doing iterative generation is as below. First you have to initialize an empty cache of the type you want, and you can start feeding in new prompts iteratively. Keeping track of dialogues history and formatting can be done with chat templates, read more on that in [chat_templating](./chat_templating)

In case you are using Sink Cache, you have to crop your inputs to that maximum length because Sink Cache can generate text longer than its maximum window size, but it expects the first input to not exceed the maximum cache length.


```python
>>> import torch
>>> from transformers import AutoTokenizer,AutoModelForCausalLM
>>> from transformers.cache_utils import (
>>>     DynamicCache,
>>>     SinkCache,
>>>     StaticCache,
>>>     SlidingWindowCache,
>>>     QuantoQuantizedCache,
>>>     QuantizedCacheConfig,
>>> )

>>> model_id = "meta-llama/Llama-2-7b-chat-hf"
>>> model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
>>> tokenizer = AutoTokenizer.from_pretrained(model_id)

>>> user_prompts = ["Hello, what's your name?", "Btw, yesterday I was on a rock concert."]

>>> past_key_values = DynamicCache()
>>> max_cache_length = past_key_values.get_max_length()

>>> messages = []
>>> for prompt in user_prompts:
...     messages.append({"role": "user", "content": prompt})
...     inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
...     if isinstance(past_key_values, SinkCache):
...         inputs = {k: v[:, -max_cache_length:] for k, v in inputs.items()}
...
...     input_length = inputs["input_ids"].shape[1]
...
...     outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=past_key_values)
...     completion = tokenizer.decode(outputs[0, input_length: ], skip_special_tokens=True)
...     messages.append({"role": "assistant", "content": completion})

print(messages)
[{'role': 'user', 'content': "Hello, what's your name?"}, {'role': 'assistant', 'content': " Hello! My name is LLaMA, I'm a large language model trained by a team of researcher at Meta AI. 😊"}, {'role': 'user', 'content': 'Btw, yesterday I was on a rock concert.'}, {'role': 'assistant', 'content': ' Oh, cool! That sounds like a lot of fun! 🎉 Did you enjoy the concert? What was the band like? 🤔'}]
```


## Re-use Cache to continue generation

Sometimes you would want to first fill-in cache object with key/values for certain prefix prompt and re-use it several times to generate different sequences from it. In that case you can construct a `Cache` object that will hold the instruction prompt, and re-use it several times with different text sequences.

```python
>>> import copy
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

>>> model_id = "meta-llama/Llama-2-7b-chat-hf"
>>> model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
>>> tokenizer = AutoTokenizer.from_pretrained(model_id)

>>> # Init StaticCache with big enough max-length (1024 tokens for the below example)
>>> # You can also init a DynamicCache, if that suits you better
>>> prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=1024, device="cuda", dtype=torch.bfloat16)

>>> INITIAL_PROMPT = "You are a helpful assistant. "
>>> inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to("cuda")
>>> # This is the common prompt cached, we need to run forward without grad to be abel to copy
>>> with torch.no_grad():
...      prompt_cache = model(**inputs_initial_prompt, past_key_values = prompt_cache).past_key_values

>>> prompts = ["Help me to write a blogpost about travelling.", "What is the capital of France?"]
>>> responses = []
>>> for prompt in prompts:
...     new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to("cuda")
...     past_key_values = copy.deepcopy(prompt_cache)
...     outputs = model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=20)
...     response = tokenizer.batch_decode(outputs)[0]
...     responses.append(response)

>>> print(responses)
['<s> You are a helpful assistant. Help me to write a blogpost about travelling.\n\nTitle: The Ultimate Guide to Travelling: Tips, Tricks, and', '<s> You are a helpful assistant. What is the capital of France?\n\nYes, the capital of France is Paris.</s>']
```


## Legacy cache format

Prior to the introduction of the `Cache` object, the cache of LLMs used to be a tuple of tuples of tensors. The legacy
format has a dynamic size, growing as we generate text -- very similar to `DynamicCache`. If your project depend on
this legacy format, you can seamlessly convert it to a `DynamicCache` and back.

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # `return_dict_in_generate=True` is required to return the cache. `return_legacy_cache` forces the returned cache
>>> # to be of the legacy type
>>> generation_outputs = model.generate(**inputs, return_dict_in_generate=True, return_legacy_cache=True, max_new_tokens=5)

>>> # We can convert a legacy cache to a DynamicCache -- and the other way around. This is helpful if you have custom
>>> # logic to manipulate a cache in a specific format.
>>> cache = DynamicCache.from_legacy_cache(generation_outputs.past_key_values)
>>> legacy_format_cache = cache.to_legacy_cache()
```
