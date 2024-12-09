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

Mixed precision accelerates training speed by performing some calculations in half-precision and some in full-precision. The half-precision calculations boosts training speed because it's not as computationally expensive as performing the calculations in full-precision. Preserving some of the calculations in full-precision maintains accuracy.

### Optimizers

### Data preloading

## PyTorch

### torch.empty_cache_steps

### torch.compile

### PyTorch scaled dot production attention

## PEFT
