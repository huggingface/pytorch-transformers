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

# Optimize inference

Inference with large language models (LLMs) can be challenging because they have to store and handle billions of parameters. To load a 70B parameter [Llama 2](https://hf.co/meta-llama/Llama-2-70b-hf) model, it requires 256GB of memory for full precision weights and 128GB of memory for half-precision weights. For comparison, the most powerful GPUs today - the A100 and H100 - only have 80GB of memory.

On top of the memory requirements, inference is slow because LLMs are called repeatedly to generate the next token. The input sequence increases as generation progresses, which takes longer and longer to process.

This guide will show you how to optimize LLM inference to accelerate generation and reduce memory usage.

> [!TIP]
> Try out [Text Generation Inference (TGI)](https://hf.co/docs/text-generation-inference), a Hugging Face library dedicated to deploying and serving highly optimized LLMs for inference. It includes deployment-oriented optimization features not included in Transformers, such as continuous batching for increasing throughput and tensor parallelism for multi-GPU inference.

## Static kv-cache and torch.compile

LLMs compute key-value (kv) values for each input token, and it performs the same kv computation each time because the generated output becomes part of the input. However, performing the same kv computation every time is not very efficient.

A *kv-cache* stores the past keys and values instead of recomputing them each time. But the kv-cache is dynamic and it grows with each generation step which prevents you from taking advantage of [torch.compile](./perf_torch_compile), a powerful optimization method that fuses PyTorch code into optimized kernels.

The *static kv-cache* solves this issue by pre-allocating the kv-cache size to a maximum value, which allows you to combine it with [torch.compile](./perf_torch_compile) for up to a 4x speed up. Your speed up may vary depending on the model size (larger models have a smaller speed up) and hardware.

> [!WARNING]
> Follow this [issue](https://github.com/huggingface/transformers/issues/28981) to track which models (Llama, Gemma, Mistral, etc.) support a static kv-cache and torch.compile.

Depending on your task, there are several ways you can use the static kv-cache.

1. For basic use cases, set [cache_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) to `"static"` (recommended).
2. For multi-turn generation or a custom generation loop, initialize and handle [`StaticCache`] directly.
3. For more unique hardware or use cases, it may be better to compile the entire [`~GenerationMixin.generate`] function into a single graph.

> [!TIP]
> Regardless of how you use the static kv-cache and torch.compile, left-pad your inputs with [pad_to_multiple_of](https://hf.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of) to a limited set of values to avoid shape-related recompilations.

<hfoptions id="static-kv">
<hfoption id="1. cache_implementation">

1. Set the [cache_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) to `"static"` in a models [`GenerationConfig`].
2. Call [torch.compile](./perf_torch_compile) to compile the forward pass with the static kv-cache.

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

Under the hood, [`~GenerationMixin.generate`] attempts to reuse the same cache object to avoid recompilation at each call, which is critical to get the most out of [torch.compile](./perf_torch_compile). Be aware of the following to avoid triggering recompilation or if generation is slower than expected.

1. If the batch size changes or the maximum output length increases between calls, the cache is reinitialized and recompiled.
2. The first several calls of the compiled function are slower because it is being compiled.

</hfoption>
<hfoption id="2. StaticCache">

Directly initialize a [`StaticCache`] object and pass it to the `past_key_values` parameter in [`~GenerationMixin.generate`]. The [`StaticCache`] keeps the cache contents, so you can pass it to a new [`~GenerationMixin.generate`] call to continue generation, similar to a dynamic cache.

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
> To reuse [`StaticCache`] on a new prompt, use [`~StaticCache.reset`] to reset the cache contents between calls.

Another option for using [`StaticCache`] is to pass it to a models forward pass using the same `past_key_values` argument. This allows you to write your own custom decoding function to decode the next token given the current token, position, and cache position of previously generated tokens.

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

To enable static kv-cache and [torch.compile](./perf_torch_compile) with [`StaticCache`], follow the steps below.

1. Initialize [`StaticCache`] before using the model for inference to configure parameters like the maximum batch size and sequence length.
2. Call [torch.compile](./perf_torch_compile) on the model to compile the forward pass with the static kv-cache.
3. Set `enable_math=True` in the [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) context manager to enable the native PyTorch C++ implementation of scaled dot product attention to speed up inference even more.

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
<hfoption id="3. compile entire generate function">

Compiling the entire [`~GenerationMixin.generate`] function also compiles the input preparation logit processor operations, and more in addition to the forward pass. With this approach, you don't need to initialize [`StaticCache`] or set the [cache_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) parameter.

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

This usage pattern is more appropriate for unique hardware or use cases, but there are several drawbacks to consider.

1. Compilation is much slower.
2. Parameters must be configured through [`GenerationConfig`].
3. Many warnings and exceptions are suppressed. We recommend testing the uncompiled model first.
4. Many features are unavailable at the moment. For example, generation does not stop if an `EOS` token is selected.

</hfoption>
</hfoptions>

## Decoding

Decoding can also be optimized to accelerate generation. You can use a lightweight assistant model to generate candidate tokens faster than the LLM itself or you can use a variant of this decoding strategy that works especially well for input-grounded tasks.

### Speculative decoding

> [!TIP]
> For a more in-depth explanation, take a look at the [Assisted Generation: a new direction toward low-latency text generation](https://hf.co/blog/assisted-generation) blog post!

For each input token, the model weights are loaded each time during the forward pass, which is slow and cumbersome when a model has billions of parameters. Speculative decoding alleviates this slowdown by using a second smaller and faster assistant model to generate candidate tokens that are verified by the larger model in a single forward pass. If the verified tokens are correct, the LLM essentially gets them for "free" without having to generate them itself. There is no degradation in accuracy because the verification forward pass ensures the same outputs are generated as if the LLM had generated them on its own.

To get the largest speed up, the assistant model should be a lot smaller than the LLM so that it can generate tokens quickly. The assistant and LLM model must also share the same tokenizer to avoid re-encoding and decoding tokens.

> [!WARNING]
> Speculative decoding is only supported for the greedy search and sampling decoding strategies, and it also doesn't support batched inputs.

Enable speculative decoding by loading an assistant model and passing it to [`~GenerationMixin.generate`].

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

For speculative sampling decoding, add the [do_sample](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.do_sample) and [temperature](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.temperature) parameters to [`~GenerationMixin.generate`].

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

To enable prompt lookup decoding, specify the number of tokens that should be overlapping in the [prompt_lookup_num_tokens](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.prompt_lookup_num_tokens) parameter. Then pass this parameter to [`~GenerationMixin.generate`].

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

For prompt lookup decoding with sampling, add the [do_sample](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.do_sample) and [temperature](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.temperature) parameters to [`~GenerationMixin.generate`].

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

## Attention

A known issue with transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this, try FlashAttention2 or PyTorch's scaled dot product attention (SDPA), which are more memory efficient attention implementations.

### FlashAttention-2

FlashAttention and [FlashAttention-2](./perf_infer_gpu_one#flashattention-2) break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to the GPU memory to speed up inference. FlashAttention-2 improves on the original FlashAttention algorithm by also parallelizing over sequence length dimension and better partitioning work on the hardware to reduce synchronization and communication overhead.

To use FlashAttention-2, set [attn_implementation](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.PreTrainedModel.from_pretrained.attn_implementation) to `"flash_attention_2"` in [`~PreTrainedModel.from_pretrained`].

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
> SDPA automaticallysupports FlashAttention-2 as long as you have the latest PyTorch version installed.

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

Quantization reduces the size of model weights by storing them in a lower precision. This translates to lower memory usage and makes loading LLMs for inference more accessible if you're constrained by GPU memory. If you aren't limited by your GPU, you don't necessarily need to quantize your model because it can increase latency slightly (except for AWQ and fused AWQ modules) due to the extra step required to quantize and dequantize the weights.

> [!TIP]
> There are many quantization libraries (see the [Quantization](./quantization) guide for more details) available, such as Quanto, AQLM, VPTQ, AWQ, and AutoGPTQ. Feel free to try them out and see which one works best for your use case. We also recommend reading the [Overview of natively supported quantization schemes in 🤗 Transformers](https://hf.co/blog/overview-quantization-transformers) blog post which compares AutoGPTQ and bitsandbytes.

Use the Model Memory Calculator below to estimate and compare how much memory is required to load a model. For example, try estimating the memory required to load [Mistral-7B-v0.1](https://hf.co/mistralai/Mistral-7B-v0.1).

<iframe
	src="https://hf-accelerate-model-memory-usage.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

To load a model in half-precision, set the [torch_dtype](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.PreTrainedModel.from_pretrained.torch_dtype) parameter in [`~transformers.AutoModelForCausalLM.from_pretrained`] to `torch.bfloat16`. This requires 13.74GB of memory.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto",
)
```

To load a quantized model (8-bit or 4-bit), try [bitsandbytes](https://hf.co/docs/bitsandbytes) and set the [load_in_4bit](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.BitsAndBytesConfig.load_in_4bit) or [load_in_8bit](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.BitsAndBytesConfig.load_in_8bit) parameters to `True`. Loading the model in 8-bits only requires 6.87 GB of memory.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config, device_map="auto"
)
```
