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

# LLM inference optimization

Large language models (LLMs) have pushed text generation applications, such as chat and code completion models, to the next level by producing text that displays a high level of understanding and fluency. But what makes LLMs so powerful - namely their size - also presents challenges for inference.

Basic inference is slow because LLMs have to be called repeatedly to generate the next token. The input sequence increases as generation progresses, which takes longer and longer for the LLM to process. LLMs also have billions of parameters, making it a challenge to store and handle all those weights in memory.

This guide will show you how to use the optimization techniques available in Transformers to accelerate LLM inference.

> [!TIP]
> Hugging Face also provides [Text Generation Inference (TGI)](https://hf.co/docs/text-generation-inference), a library dedicated to deploying and serving highly optimized LLMs for inference. It includes deployment-oriented optimization features not included in Transformers, such as continuous batching for increasing throughput and tensor parallelism for multi-GPU inference.

## Static kv-cache and `torch.compile`

During decoding, a LLM computes the key-value (kv) values for each input token and since it is autoregressive, it computes the same kv values each time because the generated output becomes part of the input now. This is not very efficient because you're recomputing the same kv values each time.

To optimize this, you can use a kv-cache to store the past keys and values instead of recomputing them each time. However, since the kv-cache grows with each generation step and is dynamic, it prevents you from taking advantage of [`torch.compile`](./perf_torch_compile), a powerful optimization tool that fuses PyTorch code into fast and optimized kernels. We have an entire guide dedicated to kv-caches [here](./kv_cache).

The *static kv-cache* solves this issue by pre-allocating the kv-cache size to a maximum value which allows you to combine it with `torch.compile` for up to a 4x speed up. Your speed up may vary depending on the model size (larger models have a smaller speed up) and hardware.

> [!WARNING]
> Currently, only [Llama](./model_doc/llama2) and a few other models support static kv-cache and `torch.compile`. Check [this issue](https://github.com/huggingface/transformers/issues/28981) for a live model compatibility list.

There are three flavors of static kv-cache usage, depending on the complexity of your task:
1. Basic usage: simply set a flag in `generation_config` (recommended);
2. Advanced usage: handle a cache object for multi-turn generation or a custom generation loop;
3. Advanced usage: compile the entire `generate` function into a single graph, if having a single graph is relevant for you.

Select the correct tab below for further instructions on each of these flavors.

> [!TIP]
> Regardless of the strategy used with `torch.compile`, you can avoid shape-related recompilations if you left-pad your LLM inputs to a limited set of values. The [`pad_to_multiple_of` tokenizer flag](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of) is your friend!

<hfoptions id="static-kv">
<hfoption id="basic usage: generation_config">

For this example, let's use the [Gemma](https://hf.co/google/gemma-2b) model. All we need to do is to:
1. Access the model's `generation_config` attribute and set the `cache_implementation` to "static";
2. Call `torch.compile` on the model to compile the forward pass with the static kv-cache.

And that's it!

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype="auto", device_map="auto")

model.generation_config.cache_implementation = "static"

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)

outputs = model.generate(**input_ids)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

Under the hood, `generate` will attempt to reuse the same cache object, removing the need for re-compilation at each call. Avoiding re-compilation is critical to get the most out of `torch.compile`, and you should be aware of the following:
1. If the batch size changes or the maximum output length increases between calls, the cache will have to be reinitialized, triggering a new compilation;
2. The first couple of calls of the compiled function are slower, as the function is being compiled.

> [!WARNING]
> For a more advanced usage of the static cache, such as multi-turn conversations, we recommend instantiating and manipulating the cache object outside [`~GenerationMixin.generate`]. See the advanced usage tab.

</hfoption>
<hfoption id="advanced usage: control Static Cache">

A [`StaticCache`] object can be passed to the model's [`~GenerationMixin.generate`] under the `past_key_values` argument. The object will retain the cache contents, so you can pass it to a new [`~GenerationMixin.generate`] call to continue generation, like you would do with a dynamic cache.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype="auto", device_map="auto")

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)
prompt_length = input_ids.input_ids.shape[1]
model.generation_config.max_new_tokens = 16

past_key_values = StaticCache(
    config=model.config,
    batch_size=1,
    # If you plan to reuse the cache, make sure the cache length is large enough for all cases
    max_cache_len=prompt_length+(model.generation_config.max_new_tokens*2),
    device=model.device,
    dtype=model.dtype
)
outputs = model.generate(**input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference frames. 2']

# pass in the generated text and the same cache object to continue generation from where it left off. Optionally, in a
# multi-turn conversation, append the new user input to the generated text.
new_input_ids = outputs
outputs = model.generate(new_input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference frames. 2. The speed of light is constant in all inertial reference frames. 3.']
```

> [!TIP]
> If you want to reuse the same [`StaticCache`] object on a new prompt, be sure to reset its contents with the `.reset()` method between calls

If you want to go further down a level, the [`StaticCache`] object can also be passed to the model's forward pass under the same `past_key_values` argument. Using this strategy, you can write your own function to decode the next token given the current token and position and cache position of previously generated tokens.

```py
from transformers import LlamaTokenizer, LlamaForCausalLM, StaticCache, logging
from transformers.testing_utils import CaptureLogger
import torch
from accelerate.test_utils.testing import get_backend

prompts = [
    "Simply put, the theory of relativity states that ",
    "My favorite all time favorite condiment is ketchup.",
]

NUM_TOKENS_TO_GENERATE = 40
torch_device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="sequential")
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token
```

There are a few important things you must do to enable static kv-cache and `torch.compile` with the `StaticCache` method:
1. Initialize the [`StaticCache`] instance before using the model for inference. There you can configure parameters like the maximum batch size and sequence length.
2. Call `torch.compile` on the model to compile the forward pass with the static kv-cache.
3. Use `SDPBackend.MATH` in the [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) context manager to enable the native PyTorch C++ implementation of scaled dot product attention to speed up inference even more.

```py
from torch.nn.attention import SDPBackend, sdpa_kernel

batch_size, seq_length = inputs["input_ids"].shape
with torch.no_grad():
    past_key_values = StaticCache(
        config=model.config, batch_size=2, max_cache_len=4096, device=torch_device, dtype=model.dtype
    )
    cache_position = torch.arange(seq_length, device=torch_device)
    generated_ids = torch.zeros(
        batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=torch.int, device=torch_device
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch_device).to(torch.int)

    logits = model(
        **inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True
    )[0]
    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids[:, seq_length] = next_token[:, 0]

    decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
    cache_position = torch.tensor([seq_length + 1], device=torch_device)
    for _ in range(1, NUM_TOKENS_TO_GENERATE):
        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_tokens(model, next_token.clone(), None, cache_position, past_key_values)
            generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
text
['Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the same for all observers, and 3) the laws of physics are the same for all observers.',
 'My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p']
```

</hfoption>
<hfoption id="advanced usage: end-to-end generate compilation">

Compiling the entire `generate` function, in terms of code, is even simpler than in the basic usage: call `torch.compile` on `generate` to compile the entire function. No need to specify the use of the static cache: although it is compatible, dynamic cache (default) was faster in our benchmarks.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype="auto", device_map="auto")

model.generate = torch.compile(model.generate, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)

outputs = model.generate(**input_ids)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

As a result, we compile not only the model forward pass, but also all input preparation, logit processor operations, and so on. The result should be a slightly `generate` call, compared to the basic usage example, and the compiled graph may be better suited to more exotic hardware devices or use cases. However, there are severe drawbacks in using this approach:
1. Compilation is much slower;
2. All parameterization of `generate` must be done through `generation_config`;
3. Many warnings and exceptions are suppressed -- we suggest testing with its uncompiled form first;
4. Although we are working on it, it is heavily feature restricted (for instance, at the time of writing, generation does not stop if an EOS token is selected).

</hfoption>
</hfoptions>

## Speculative decoding

> [!TIP]
> For a more in-depth explanation, take a look at the [Assisted Generation: a new direction toward low-latency text generation](https://hf.co/blog/assisted-generation) blog post!

Another issue with autoregression is that for each input token you need to load the model weights each time during the forward pass. This is slow and cumbersome for LLMs which have billions of parameters. Speculative decoding alleviates this slowdown by using a second smaller and faster assistant model to generate candidate tokens that are verified by the larger LLM in a single forward pass. If the verified tokens are correct, the LLM essentially gets them for "free" without having to generate them itself. There is no degradation in accuracy because the verification forward pass ensures the same outputs are generated as if the LLM had generated them on its own.

To get the largest speed up, the assistant model should be a lot smaller than the LLM so that it can generate tokens quickly. The assistant and LLM model must also share the same tokenizer to avoid re-encoding and decoding tokens.

> [!WARNING]
> Speculative decoding is only supported for the greedy search and sampling decoding strategies, and it also doesn't support batched inputs.

Enable speculative decoding by loading an assistant model and passing it to the [`~GenerationMixin.generate`] method.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype="auto").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Einstein's theory of relativity states that the speed of light is constant.    "]
```

</hfoption>
<hfoption id="sampling">

For speculative sampling decoding, add the `do_sample` and `temperature` parameters to the [`~GenerationMixin.generate`] method in addition to the assistant model.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype="auto").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
["Einstein's theory of relativity states that motion in the universe is not a straight line.\n"]
```

</hfoption>
</hfoptions>

### Prompt lookup decoding

Prompt lookup decoding is a variant of speculative decoding that is also compatible with greedy search and sampling. Prompt lookup works especially well for input-grounded tasks - such as summarization - where there is often overlapping words between the prompt and output. These overlapping n-grams are used as the LLM candidate tokens.

To enable prompt lookup decoding, specify the number of tokens that should be overlapping in the `prompt_lookup_num_tokens` parameter. Then you can pass this parameter to the [`~GenerationMixin.generate`] method.

<hfoptions id="pld">
<hfoption id="greedy decoding">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype="auto").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The second law of thermodynamics states that entropy increases with temperature.      ']
```

</hfoption>
<hfoption id="sampling">

For prompt lookup decoding with sampling, add the `do_sample` and `temperature` parameters to the [`~GenerationMixin.generate`] method.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype="auto").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3, do_sample=True, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
["The second law of thermodynamics states that energy cannot be created nor destroyed. It's not a"]
```

</hfoption>
</hfoptions>

## Attention optimizations

A known issue with transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this, try FlashAttention2 or PyTorch's scaled dot product attention (SDPA), which are more memory efficient attention implementations and can accelerate inference.

### FlashAttention-2

FlashAttention and [FlashAttention-2](./perf_infer_gpu_one#flashattention-2) break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to GPU memory to speed up inference. FlashAttention-2 improves on the original FlashAttention algorithm by also parallelizing over sequence length dimension and better partitioning work on the hardware to reduce synchronization and communication overhead.

To use FlashAttention-2, set `attn_implementation="flash_attention_2"` in the [`~PreTrainedModel.from_pretrained`] method.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### Fine-Tuning with torch.compile and Padding-Free Data Collation

In addition to optimizing inference, you can also enhance the training efficiency of large language models by leveraging torch.compile during fine-tuning and using a padding-free data collator. This approach can significantly speed up training and reduce computational overhead.

Here's how you can fine-tune a Llama model using SFTTrainer from the TRL library, with torch_compile enabled and a padding-free data collator:

```
#################### IMPORTS ###################

import math
import datasets
import dataclasses
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

#################### MODEL LOADING WITH FLASH ATTENTION ###################

model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"  # Enables FlashAttention-2
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

#################### DATA PREPROCESSING (PADDING-FREE) ###################

response_template = "\n### Label:"
response_template_ids = tokenizer.encode(
    response_template, add_special_tokens=False
)[2:]  # Exclude special tokens

data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids=response_template_ids,
    tokenizer=tokenizer,
    ignore_index=-100,
    padding_free=True  # Enables padding-free collation
)

def format_dataset(example):
    return {
        "output": example["output"] + tokenizer.eos_token
    }

data_files = {"train": "path/to/dataset"}  # Replace with your dataset path
json_dataset = datasets.load_dataset("json", data_files=data_files)
formatted_train_dataset = json_dataset["train"].map(format_dataset)

################# TRAINING CONFIGURATION ############################

train_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    weight_decay=0.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    include_tokens_per_second=True,
    save_strategy="epoch",
    output_dir="output",
    torch_compile=True,  # Enables torch.compile
    torch_compile_backend="inductor",
    torch_compile_mode="default"
)

# Convert TrainingArguments to SFTConfig
transformer_train_arg_fields = [x.name for x in dataclasses.fields(SFTConfig)]
transformer_kwargs = {
    k: v
    for k, v in train_args.to_dict().items()
    if k in transformer_train_arg_fields
}
training_args = SFTConfig(**transformer_kwargs)

####################### FINE-TUNING #####################

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    data_collator=data_collator,
    dataset_text_field="output",
    args=training_args,
)
trainer.train()
```

### PyTorch scaled dot product attention

Scaled dot product attention (SDPA) is automatically enabled in PyTorch 2.0 and it supports FlashAttention, xFormers, and PyTorch's C++ implementation. SDPA chooses the most performant attention algorithm if you're using a CUDA backend. For other backends, SDPA defaults to the PyTorch C++ implementation.

> [!TIP]
> SDPA supports FlashAttention-2 as long as you have the latest PyTorch version installed.

Use the [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) context manager to explicitly enable or disable any of the four attention algorithms. For example, use `SDPBackend.FLASH_ATTENTION` to enable FlashAttention.

```py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.bfloat16,
)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)
```

## Quantization

Quantization reduces the size of the LLM weights by storing them in a lower precision. This translates to lower memory usage and makes loading LLMs for inference more accessible if you're constrained by your GPUs memory. If you aren't limited by your GPU, you don't necessarily need to quantize your model because it can incur a small latency cost (except for AWQ and fused AWQ modules) due to the extra step required to quantize and dequantize the weights.

> [!TIP]
> There are many quantization libraries (see the [Quantization](./quantization) guide for more details) available, such as Quanto, AQLM, VPTQ, AWQ, and AutoGPTQ. Feel free to try them out and see which one works best for your use case. We also recommend reading the [Overview of natively supported quantization schemes in 🤗 Transformers](https://hf.co/blog/overview-quantization-transformers) blog post which compares AutoGPTQ and bitsandbytes.

Use the Model Memory Calculator below to estimate and compare how much memory is required to load a model. For example, try estimating how much memory it costs to load [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).

<iframe
	src="https://hf-accelerate-model-memory-usage.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

To load Mistral-7B-v0.1 in half-precision, set the `torch_dtype` parameter in the [`~transformers.AutoModelForCausalLM.from_pretrained`] method to `torch.bfloat16`. This requires 13.74GB of memory.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto",
)
```

To load a quantized model (8-bit or 4-bit) for inference, try [bitsandbytes](https://hf.co/docs/bitsandbytes) and set the `load_in_4bit` or `load_in_8bit` parameters to `True`. Loading the model in 8-bits only requires 6.87 GB of memory.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config, device_map="auto"
)
```
