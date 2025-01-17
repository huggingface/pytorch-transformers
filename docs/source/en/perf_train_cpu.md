<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU

A modern CPU is capable of efficiently training large models by leveraging the underlying optimizations built into the hardware and training on fp16 or bf16 datatypes.

This guide focuses on how to train large models on an Intel CPU using mixed precision and the [Intel Extension for PyTorch (IPEX)](https://intel.github.io/intel-extension-for-pytorch/index.html) library.

Install IPEX with the command below. You can find your PyTorch version by running `pip list | grep torch` in the command line.

```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

> [!TIP]
> Refer to the IPEX [installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation) guide for more details.

IPEX provides additional performance optimizations for Intel CPUs. These include additional CPU instruction level architecture (ISA) support such as [Intel AVX512-VNNI](https://en.wikichip.org/wiki/x86/avx512_vnni) and [Intel AMX](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html). Both of these features are designed to accelerate matrix multiplication. Older AMD and Intel CPUs with only Intel AVX2, however, aren't guaranteed better performance with IPEX.

IPEX also supports [Auto Mixed Precision (AMP)](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html) training with the fp16 and bf16 datatypes. Reducing precision speeds up training and reduces memory usage because it requires less computation. The loss in accuracy from using full-precision is minimal. 3rd, 4th, and 5th generation Intel Xeon Scalable processors natively support bf16, and the 6th generation processor also natively supports fp16 in addition to bf16.

AMP is enabled for CPU backends training with PyTorch.

[`Trainer`] supports AMP training with a CPU by adding the `--use_cpu`, `--use_ipex`, and `--bf16` parameters. The example below demonstrates the [run_qa.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) script.

```bash
python run_qa.py \
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
 --use_ipex \
 --bf16 \
 --use_cpu
```

These parameters can also be added to [`TrainingArguments`] as shown below.

```py
training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    use_ipex=True,
    use_cpu=True,
)
```

## Resources

Learn more about training on Intel CPUs in the [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids) blog post.
