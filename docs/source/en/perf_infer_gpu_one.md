<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPU

GPUs are the standard hardware for machine learning because they're optimized for memory bandwidth and parallelism. With the increasing sizes of modern models, it's more important than ever to make sure GPUs are capable of efficiently handling them and able to deliver the best possible performance.

This guide will demonstrate a few ways to optimize inference on a GPU. The optimization methods shown below can be combined with each other to achieve even better performance, and they also work for distributed GPUs.

## bitsandbytes

[bitsandbytes](https://hf.co/docs/bitsandbytes/index) is a quantization library that supports 8-bit and 4-bit quantization. Quantization represents weights in a lower precision compared to the original full precision format. It reduces memory requirements and makes it easier to fit large model into memory.

Make sure bitsandbytes and Accelerate are installed first.

```bash
pip install bitsandbytes accelerate
```

<hfoptions id="bnb">
<hfoption id="8-bit">

For text generation with 8-bit quantization, you should use [`~GenerationMixin.generate`] instead of the high-level [`Pipeline`] API. The [`Pipeline`] returns slower performance because it isn't optimized for 8-bit models, and some sampling strategies (nucleus sampling) also aren't supported.

Set up a [`BitsAndBytesConfg`] and set `load_in_8bit=True` to load a model in 8-bit precision. The [`BitsAndBytesConfig`] is passed to the `quantization_config` parameter in [`~PreTrainedModel.from_pretrained`].

Allow Accelerate to automatically distribute the model across your available hardware by setting [device_map="auto"](https://hf.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map).

Place all inputs on the same device as the model.

```py
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", quantization_config=quantization_config)

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

For distributed setups, use the `max_memory` parameter to create a mapping of the amount of memory to allocate to each GPU. The example below distributes 16GB of memory to the first GPU and 16GB of memory to the second GPU.

```py
max_memory_mapping = {0: "16GB", 1: "16GB"}
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", device_map="auto", quantization_config=quantization_config, max_memory=max_memory_mapping
)
```

Learn in more detail the concepts underlying 8-bit quantization in the [Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://hf.co/blog/hf-bitsandbytes-integration) blog post.

</hfoption>
<hfoption id="4-bit">

Set up a [`BitsAndBytesConfg`] and set `load_in_4bit=True` to load a model in 4-bit precision. The [`BitsAndBytesConfig`] is passed to the `quantization_config` parameter in [`~PreTrainedModel.from_pretrained`].

Allow Accelerate to automatically distribute the model across your available hardware by setting device_map=“auto”.

Place all inputs on the same device as the model.

```py
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", quantization_config=quantization_config)

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

For distributed setups, use the `max_memory` parameter to create a mapping of the amount of memory to allocate to each GPU. The example below distributes 16GB of memory to the first GPU and 16GB of memory to the second GPU.

```py
max_memory_mapping = {0: "16GB", 1: "16GB"}
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", device_map="auto", quantization_config=quantization_config, max_memory=max_memory_mapping
)
```

</hfoption>
</hfoptions>

## Optimum

[Optimum](https://hf.co/docs/optimum/en/index) is a Hugging Face library focused on optimizing model performance across various hardware. It supports [ONNX Runtime](https://onnxruntime.ai/docs/) (ORT), a model accelerator, for a wide range of hardware and frameworks including NVIDIA GPUs and AMD GPUs that use the [ROCm](https://www.amd.com/en/products/software/rocm.html) stack.

ORT uses optimization techniques that fuse common operations into a single node and constant folding to reduce the number of computations. ORT also places the most computationally intensive operations on the GPU and the rest on the CPU to intelligently distribute the workload between the two devices.

Optimum provides the [`~optimum.onnxruntime.ORTModel`] class for loading ONNX models. Set the `provider` parameter according to the table below.

| provider | hardware |
|---|---|
| [CUDAExecutionProvider](https://hf.co/docs/optimum/main/en/onnxruntime/usage_guides/gpu#cudaexecutionprovider) | CUDA-enabled GPUs |
| [ROCMExecutionProvider](https://hf.co/docs/optimum/onnxruntime/usage_guides/amdgpu) | AMD Instinct, Radeon Pro, Radeon GPUs |
| [TensorrtExecutionProvider](https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider) | TensorRT |

For example, load the [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://hf.co/optimum/roberta-base-squad2) checkpoint for sequence classification. This checkpoint contains a [model.onnx](https://hf.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/main/onnx/model.onnx) file. If a checkpoint doesn't have a `model.onnx` file, set `export=True` to convert a checkpoint on the fly to the ONNX format.

```py
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#   export=True,
  provider="CUDAExecutionProvider",
)
```

Now you can use the model for inference in a [`Pipeline`].

```py
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
pipeline = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Both the music and visual were astounding, not to mention the actors performance.")
```

Learn more details about using ORT with Optimum in the [Accelerated inference on NVIDIA GPUs](https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#accelerated-inference-on-nvidia-gpus) and [Accelerated inference on AMD GPUs](https://hf.co/docs/optimum/onnxruntime/usage_guides/amdgpu#accelerated-inference-on-amd-gpus) guides.

### BetterTransformer

[BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) is a *fastpath* execution of specialized Transformers functions directly on the hardware level such as a GPU. There are two main components of the fastpath execution.

- fusing multiple operations into a single kernel for faster and more efficient execution
- skipping unnecessary computation of padding tokens with nested tensors

> [!WARNING]
> Some BetterTransformer features are being upstreamed to Transformers with default support for native [torch.nn.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA). BetterTransformer has than the Transformers SDPA integration, but you can expect more and more architectures to natively support SDPA in Transformers.

BetterTransformer is available through Optimum with the [`~PreTrainedModel.to_bettertransformer`] method.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")
model = model.to_bettertransformer()
```

Call [`~PreTrainedModel.reverse_bettertransformer`] and save it first to return the model to the original Transformers model.

```py
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

Refer to the benchmarks in [Out of the box acceleration and memory savings of 🤗 decoder models with PyTorch 2.0](https://pytorch.org/blog/out-of-the-box-acceleration/) for BetterTransformer and scaled dot product attention performance. The [BetterTransformer](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) blog post also discusses fastpath execution in greater detail if you're interested in learning more.

## Scaled dot product attention (SDPA)

PyTorch's [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) is a native implementation of the scaled dot product attention mechanism. SDPA is a more efficient and optimized version of the attention mechanism used in transformer models.

There are three supported implementations available.

- [FlashAttention2](https://github.com/Dao-AILab/flash-attention) only supports models with the fp16 or bf16 torch type. Make sure to cast your model to the appropriate type first.
- [xFormers](https://github.com/facebookresearch/xformers) or Memory-Efficient Attention is able to support models with the fp32 torch type.
- C++ implementation of scaled dot product attention

SDPA is used by default for PyTorch v2.1.1. and greater when an implementation is available. You could explicitly enable SDPA by setting `attn_implementation="sdpa"` in [`~PreTrainedModel.from_pretrained`] though. Certain attention parameters, such as `head_mask` and `output_attentions=True`, are unsupported and returns a warning that Transformers will fall back to the (slower) eager implementation.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", attn_implementation="sdpa")
```

SDPA selects the most performant implementation available, but you can also explicitly select an implementation with [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) as a context manager. The example below shows how to enable the FlashAttention2 implementation with `enable_flash=True`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto").to("cuda")

input_text = "Hello, my llama is cute"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you encounter the following `RuntimeError`, try installing the nightly version of PyTorch which has broader coverage for FlashAttention.

```bash
RuntimeError: No available kernel. Aborting execution.

pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

## FlashAttention

[FlashAttention](https://github.com/Dao-AILab/flash-attention) is also available as a standalone package. It can significantly speed up inference by:

1. additionally parallelizing the attention computation over sequence length
2. partitioning the work between GPU threads to reduce communication and shared memory reads/writes between them

Install FlashAttention first for the hardware you're using.

<hfoptions id="install">
<hfoption id="NVIDIA">

```bash
pip install flash-attn --no-build-isolation
```

</hfoption>
<hfoption id="AMD">

FlashAttention2 support is currently limited to Instinct MI210, Instinct MI250 and Instinct MI300. We strongly suggest running this [Dockerfile](https://github.com/huggingface/optimum-amd/tree/main/docker/transformers-pytorch-amd-gpu-flash/Dockerfile) for FlashAttention2 on AMD GPUs.

</hfoption>
</hfoptions>

Enable FlashAttention2 by setting `attn_implementation="flash_attention_2"` in [`~PreTrainedModel.from_pretrained`]. FlashAttention2 is only supported for models with the fp16 or bf16 torch type. Make sure to cast your model to the appropriate data type first.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
```

### Benchmark results

FlashAttention2 speeds up inference considerably especially for inputs with long sequences. However, since FlashAttention2 doesn't support computing attention scores with padding tokens, you must manually pad and unpad the attention scores for batched inference if a sequence contains padding tokens. The downside is batched generation is slower with padding tokens. 

<hfoptions id="padded">
<hfoption id="short sequence length">

With a relatively small sequence length, a single forward pass creates overhead leading to a small speed up. The graph below shows the expected speed up for a single forward pass with [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) with padding.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png"/>
</div>

</hfoption>
<hfoption id="long sequence length">

You can train on much longer sequence lengths without running into out-of-memory issues with FlashAttention2, and potentially reduce memory usage up to 20x. The speed up benefits are even better. The graph below shows the expected speed up for a single forward pass with [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) with padding on a longer sequence length.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-large-seqlen-padding.png"/>
</div>

</hfoption>
</hfoptions>

To avoid this slowdown, use FlashAttention2 without padding tokens in the sequence during training. Pack the dataset or concatenate sequences until reaching the maximum sequence length.

<hfoptions id="not-padded">
<hfoption id="tiiuae/falcon-7b">

The graph below shows the expected speed up for a single forward pass with [tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b) with a sequence length of 4096 and various batch sizes without padding tokens.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png"/>
</div>

</hfoption>
<hfoption id="meta-llama/Llama-7b-hf">

The graph below shows the expected speed up for a single forward pass with [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) with a sequence length of 4096 and various batch sizes without padding tokens.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png"/>
</div>

</hfoption>
</hfoptions>
