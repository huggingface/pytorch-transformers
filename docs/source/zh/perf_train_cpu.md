<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 在CPU上进行高效训练

本指南将重点介绍如何在CPU上高效训练大型模型。

## 使用IPEX进行混合精度训练
混合精度训练在模型中可以同时使用单精度（fp32）和半精度（bf16/fp16）的数据类型来加速训练或推理过程，并且仍然能保留大部分单精度的准确性。现代的CPU，例如第三代、第四代和第五代Intel® Xeon® Scalable处理器，原生支持bf16，而第六代Intel® Xeon® Scalable处理器原生支持bf16和fp16。您在训练时启用bf16或fp16的混合精度训练可以直接提高处理性能。

为了进一步最大化训练性能，您可以使用Intel® PyTorch扩展（IPEX）。IPEX是一个基于PyTorch构建的库，增加了额外的CPU指令集架构（ISA）级别的支持，比如Intel®高级向量扩展512（Intel® AVX512-VNNI）和Intel®高级矩阵扩展（Intel® AMX）。这为Intel CPU提供额外的性能提升。然而，仅支持AVX2的CPU（例如AMD或较旧的Intel CPU）在使用IPEX时并不保证能提高性能。

从PyTorch 1.10版本起，CPU后端已经启用了自动混合精度（AMP）。IPEX还支持bf16/fp16的AMP和bf16/fp16算子优化，并且部分功能已经上游到PyTorch主分支。通过IPEX AMP，您可以获得更好的性能和用户体验。

点击[这里](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html)查看**自动混合精度**的更多详细信息。


### IPEX 安装:

IPEX 的发布与 PyTorch 一致，您可以通过 pip 安装：

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 2.5.0             |  2.5.0+cpu     |
| 2.4.0             |  2.4.0+cpu     |
| 2.3.0             |  2.3.0+cpu     |
| 2.2.0             |  2.2.0+cpu     |

请运行 `pip list | grep torch` 以获取您的 `pytorch_version`，然后根据该版本安装相应的 `IPEX version_name`。
```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

如果需要的话，您可以在 [ipex-whl-stable-cpu](https://developer.intel.com/ipex-whl-stable-cpu) 查看最新版本。

查看更多 [安装IPEX](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html) 的方法。


### 在 Trainer 中使用 IPEX
在 Trainer 中使用 IPEX 时，您应在训练命令参数中添加 `use_ipex`、`bf16` 或 `fp16` 以及 `no_cuda` 来启用自动混合精度。

以 [Transformers 问答任务](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)为例：

- 在 CPU 上使用 BF16 自动混合精度训练 IPEX 的示例如下：
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

如果您想在脚本中启用 `use_ipex` 和 `bf16`，请像下面这样将这些参数添加到 `TrainingArguments` 中：
```diff
training_args = TrainingArguments(
    output_dir=args.output_path,
+   bf16=True,
+   use_ipex=True,
+   use_cpu=True,
    **kwargs
)
```

### 实践示例

博客: [使用 Intel Sapphire Rapids 加速 PyTorch Transformers](https://huggingface.co/blog/intel-sapphire-rapids)
