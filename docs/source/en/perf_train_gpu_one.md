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

GPUs are commonly used to train deep learning models due to their high memory bandwidth and parallel processing capabilities. Depending on your GPU and model size, it is possible to even train models with billions of parameters. The key is to find the right balance between GPU memory utilization (data throughput/training time) and training speed.

This guide will show you the features available in Transformers for efficiently training a model on a single GPU. In many cases, you'll want to use a combination of these features to optimize training.

Refer to the table below to quickly help you identify the features relevant to your training scenario.

| Feature | Training speed | Memory usage |
|---|---|---|
| batch size | yes | yes |
| gradient accumulation | no | yes |
| gradient checkpointing | no | yes |
| mixed precision | yes | depends |
| optimizers | yes | yes |
| data preloading | yes | no |
| torch_empty_cache_steps | no | yes |
| torch.compile | yes | no |
| PEFT | no | yes |

## Trainer

[Trainer](./trainer) supports many useful training features that can be configured through [`TrainingArguments`]. This section highlights some of the more important features for optimizing training.

### Batch size

Batch size is one of the most important hyperparameters for efficient GPU training because it affects memory usage and training speed. Larger batch sizes lead to faster training because it takes advantage of GPUs parallel processing power. It is recommended to use batch sizes that are powers of 2, such as 8, 64, 128, 256, 512, etc. The batch size depends on your GPU and the models data type.

Configure [`~TrainingArguments.per_device_train_batch_size`] in [`TrainingArguments`].

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
)
```

Refer to the NVIDIA [Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features) guide to learn more about how input features and output neuron counts and batch size affect performance. These are involved in the General Matrix Multiplications (GEMMs) performed by the GPU. Larger parameters are better for parallelization and efficiency.

The [Tensor Core Requirements](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) section is also useful for selecting a batch size that maximizes the speed of tensor multiplication based on the data type and GPU. For example, multiples of 8 are recommended for fp16, unless it's an A100 GPU, in which case use multiples of 65.

Finally, consider [Dimension Quantization Effects](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization) for smaller parameters. Tile quantization results when matrix dimensions aren't divisible by a GPUs thread block tile size, causing the GPU to underutilize its resources. Selecting the correct batch size multiplier, such that the matrix is divisible by the tile size, can significantly speed up training.

### Gradient accumulation

Gradient accumulation overcomes memory constraints - useful for fitting a very large model that otherwise wouldn't fit on a single GPU - by accumulating gradients over multiple mini-batches before updating the parameters. This reduces memory by storing fewer gradients and enables training with a larger *effective batch size* because usually, the parameters are updated from a single batch of data. Training can slow down though due to the additional forward and backward passes introduced by gradient accumulation.

Configure [`~TrainingArguments.per_device_train_batch_size`] in [`TrainingArguments`] to enable gradient accumulation.

```py
from transformers import TrainingArguments

# effective batch size of 64
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
)
```

Try to avoid too many gradient accumulation steps because it can really slow down training. Consider the example below, where the maximum batch size that'll fit on your GPU is 4. You should keep your batch size at 4 to better utilize the GPU.

| batch size | gradient accumulation steps | effective batch size |  |
|---|---|---|---|
| 1 | 64 | 64 | 👎 |
| 4 | 16 | 64 | 👍 |

### Gradient checkpointing

Gradient checkpointing reduces memory usage by only storing some of the intermediate activations during the backward pass and recomputing the remaining activations. This avoids storing *all* of the intermediate activations from the forward pass, which can require a lot of memory overhead. However, it comes at the cost of slower training speed (~20%).

Configure [`~TrainingArguments.gradient_checkpointing`] in [`TrainingArguments`] to enable gradient checkpointing.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
)
```

### Mixed precision

Mixed precision accelerates training speed by performing some calculations in half-precision (fp16) and some in full-precision (fp32). The half-precision calculations boosts training speed because it's not as computationally expensive as performing the calculations in full-precision. Meanwhile, preserving some of the calculations in full-precision maintains accuracy.

There are several data types available for mixed precision training.

<hfoptions id="mixed-precision">
<hfoption id="fp16">

The main advantage of mixed precision training is saving the activations in fp16.

Configure [`~TrainingArguments.fp16`] in [`TrainingArguments`] to enable mixed precision training with the fp16 data type.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True.
)
```

fp16 doesn't memory-optimized because the gradients that are computed in fp16 are converted back to fp32 during the optimization step. You may end up using more GPU memory, especially for small batch sizes, because there are now two versions (fp16 and fp32) of the model on the GPU.

</hfoption>
<hfoption id="bf16">

[bf16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) trades off some precision for a much larger dynamic range, which is helpful for avoiding overflow and underflow errors. You can use bf16 without adding any loss scaling methods like you would with fp16. bf16 is supported by NVIDIAs Ampere architecture or newer.

Configure [`~TrainingArguments.fp16`] in [`TrainingArguments`] to enable mixed precision training with the bf16 data type.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
)
```

</hfoption>
<hfoption id="tf32">

[tf32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) is a mode on NVIDIA Ampere GPUs that convert the convolution and matrix multiplication inputs to tf32. All other storage and operations are kept in fp32. This allows tf32 to maintain the same range as fp32, the same precision as fp16 and more precision than bf16. Combining tf32 with fp16 or bf16 mixed precision training can improve throughput by 16x.

tf32 is enabled by default on NVIDIA Ampere GPUs, but you can also add the code below to your fp32 training or inference code to explicitly enable it.

```py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Configure [`~TrainingArguments.tf32`] in [`TrainingArguments`] to enable mixed precision training with tf32 mode.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True.
    tf32=True,
)
```

</hfoption>
</hfoptions>

### Optimizers

Transformers implements the [AdamW (adamw_torch)](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch by default. But because it stores a weighted average of past gradients, it requires additional memory proportional to the number of model parameters to store the past gradients. This can be an issue when training very large models, and in such cases, you should consider choosing a different optimizer. For example, if you have [Apex](https://nvidia.github.io/apex/index.html) installed on either [NVIDIA](https://github.com/NVIDIA/apex) or [AMD](https://github.com/ROCm/apex), then using the `adamw_apex_fused` optimizer provides the fastest training for all AdamW optimizers.

Configure [`~TrainingArguments.optim`] in [`TrainingArguments`] to choose an optimizer.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit
)
```

There are many optimizers to choose from (refer to [OptimizerNames](https://github.com/huggingface/transformers/blob/34f4080ff59b1668d919a1ba9f8bc4a3a2a3f478/src/transformers/training_args.py#L145) for a full supported list) depending on your training scenario. For example, Adafactor can significantly reduce memory requirements by storing a weighted average of a row or column instead of each element in the matrix at the cost of slower convergence. Another example is using a [8-bit AdamW optimizer](https://huggingface.co/docs/bitsandbytes) from bitsandbytes to quantize optimizer states. The optimizer state is stored in a lower precision and dequantized before being used in the optimizer step.

Refer to the [optimizer](./optimizers) guide for to learn about more specialized optimizers.

### Data preloading

Data preloading loads and prepares batches of data in advance on the CPU to ensure the GPU is continuously working, reducing GPU idling and increasing utilization. There are two ways to preload data to ensure the GPU is always working.

1. Allocate pinned memory on the CPU to store the data and transfer it directly to the GPU.
2. Increase the number of CPU threads or workers to preload the data faster.

Configure [`~TrainingArguments.dataloader_pin_memory`] and [`~TrainingArguments.dataloader_num_workers`] in [`TrainingArguments`] to allocate pinned memory and increase the number of workers.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)
```

## PyTorch

PyTorch provides several features for reducing memory requirements and increasing training speed. These features can often be enabled in Transformers by only adding a few lines of code.

### torch.empty_cache_steps

The [torch.cuda.empty_cache](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache) function releases unused cached memory, which can help avoid out-of-memory (OOM) errors at the cost of ~10% slower training.

Configure [`~TrainingArguments.torch_empty_cache_steps`] in [`TrainingArguments`] to enable torch.empty_cache after a certain number of training steps.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    torch_empty_cache_steps=4,
)
```

### torch.compile

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) compiles PyTorch code into optimized kernels that significantly speed up training. This feature relies on TorchDynamo to capture PyTorch graphs with the Frame Evaluation API. The graph can be further compiled into optimized kernels for different backends.

Configure [`~TrainingArguments.torch_compile`] in [`TrainingArguments`] to enable it, and configure [`~TrainingArguments.torch_compile_backend`] to select a backend to use.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    torch_empty_cache_steps=4,
    torch_compile=True,
    torch_compile_backend="inductor"
)
```

Refer to the table below to help you choose the right backend for your training scenario.

| backend | description | goal |
|---|---|---|
| eager | uses PyTorch to run extracted GraphModule | debugging |
| aot_eager | uses PyTorch eager mode for AOTAutograd's extracted forward and backward graphs | debugging |
| inductor | uses TorchInductor with AOTAutograd and CUDA Graphs by leveraging Triton kernels | training and inference |
| nvfuser | uses nvFuser with TorchScript | training and inference |
| aot_nvfuser | uses nvFuser with AOTAutograd | training and inference |
| aot_cudagraphs | uses CUDA Graphs with AOTAutograd | training and inference |
| ofi | uses TorchScripts [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch-jit-optimize-for-inference) | inference |
| fx2trt | uses [Torch-TensorRT](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html) | inference |
| onnxrt | uses [ONNX-RT](https://onnxruntime.ai/) for CPU and GPU inference | inference |
| ipex | uses [IPEX](https://github.com/intel/intel-extension-for-pytorch) for CPU inference | inference |

### PyTorch scaled dot production attention

PyTorch's [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) is a native implementation of the scaled dot product attention mechanism. SDPA is more efficient and optimized than the original attention mechanism in transformer models. It supports three types of scaled dot product attention.

- [FlashAttention2](https://github.com/Dao-AILab/flash-attention) is automatically enabled for models with the fp16 or bf16 torch type. Make sure to cast your model to the appropriate type first.
- [xFormers](https://github.com/facebookresearch/xformers) or Memory-Efficient Attention supports models with the fp32 torch type.
- C++ implementation of scaled dot product attention.

SDPA is enabled by default for PyTorch 2.1.1+, but it can be explicitly enabled by setting `attn_implementation="sdpa"` in [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", attn_implementation="sdpa")
```

## PEFT

[PEFT](https://huggingface.co/docs/peft/index), a library of parameter-efficient finetuning methods, enable training and storing large models often on consumer GPUs by only finetuning a small number of extra model parameters on top of the pretrained model. A significant amount of memory is saved because the GPU doesn't need to store the optimizer states and gradients for the pretrained base model.

[Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a very common PEFT method that decomposes the weight matrix into two smaller trainable matrices. Refer to the PEFT [Quicktour](https://huggingface.co/docs/peft/quicktour) for more details, but the example below demonstrates how to create a LoRA adapter for training.

```py
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

# create LoRA configuration object
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)

# create a LoRA adapter
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
model = get_peft_model(model, peft_config)
# print the number of parameters you're actually training
model.print_trainable_parameters
```

The model is ready to be passed to [`Trainer`] for training.
