<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Training on CPU

This guide focuses on training large models efficiently on CPU.

## Mixed precision with IPEX
Mixed precision uses single (fp32) and half-precision (bf16/fp16) data types in a model to accelerate training or inference while still preserving much of the single-precision accuracy. Modern CPUs such as 3rd, 4th, and 5th Gen Intel® Xeon® Scalable processors natively support bf16. 6th Gen Intel® Xeon® Scalable processors natively support bf16 and fp16. You should get more performance out of the box by enabling mixed precision training with bf16 or fp16.

To further maximize training performance, you can use Intel® Extension for PyTorch (IPEX), which is a library built on PyTorch and adds additional CPU instruction level architecture (ISA) level support such as Intel® Advanced Vector Extensions 512 Vector Neural Network Instructions (Intel® AVX512-VNNI), and Intel® Advanced Matrix Extensions (Intel® AMX) for an extra performance boost on Intel CPUs. However, CPUs with only AVX2 (e.g., AMD or older Intel CPUs) are not guaranteed to have better performance under IPEX.

Auto Mixed Precision (AMP) for CPU backends has been enabled since PyTorch 1.10. AMP support for bf16/fp16 on CPUs and bf16/fp16 operator optimization is also supported in IPEX and partially upstreamed to the main PyTorch branch. You can get better performance and user experience with IPEX AMP.

Check more detailed information for [Auto Mixed Precision](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html).

### IPEX installation:

IPEX release is following PyTorch, to install via pip:

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 2.5.0             |  2.5.0+cpu     |
| 2.4.0             |  2.4.0+cpu     |
| 2.3.0             |  2.3.0+cpu     |
| 2.2.0             |  2.2.0+cpu     |

Please run `pip list | grep torch` to get your `pytorch_version`, so you can get the `IPEX version_name`.
```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```
You can check the latest versions in [ipex-whl-stable-cpu](https://developer.intel.com/ipex-whl-stable-cpu) if needed.

Check more approaches for [IPEX installation](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).

### Usage in Trainer
To enable auto mixed precision with IPEX in Trainer, users should add `use_ipex`, `bf16` or `fp16`, and `no_cuda` in training command arguments.

Take an example of the use cases on [Transformers question-answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

- Training with IPEX using BF16 auto mixed precision on CPU:
<pre> python examples/pytorch/question-answering/run_qa.py \
--model_name_or_path google-bert/bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex</b> \
<b>--bf16</b> \
<b>--use_cpu</b></pre> 

If you want to enable `use_ipex` and `bf16` in your script, add these parameters to `TrainingArguments` like this:
```diff
training_args = TrainingArguments(
    output_dir=args.output_path,
+   bf16=True,
+   use_ipex=True,
+   use_cpu=True,
    **kwargs
)
```

### Practice example

Blog: [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids)
