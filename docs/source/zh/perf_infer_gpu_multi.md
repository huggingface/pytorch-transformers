<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 多GPU推理

某些模型现已支持内置的**张量并行**（Tensor Parallelism, TP），并通过 PyTorch 实现。张量并行技术将模型切分到多个 GPU 上，从而支持更大的模型尺寸，并对诸如矩阵乘法等计算任务进行并行化。

要启用张量并行，只需在调用 [`~AutoModelForCausalLM.from_pretrained`] 时传递参数 `tp_plan="auto"`：

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 初始化分布式环境
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

# 获取支持张量并行的模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",
)

# 准备输入tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 分布式运行
outputs = model(inputs)
```

您可以使用 `torchrun` 命令启动上述脚本，多进程模式会自动将每个进程映射到一张 GPU：

```
torchrun --nproc-per-node 4 demo.py
```

目前，PyTorch 张量并行支持以下模型：
* [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)

如果您希望对其他模型添加张量并行支持，可以通过提交 GitHub Issue 或 Pull Request 来提出请求。

### 预期性能提升

对于推理场景（尤其是处理大批量或长序列的输入），张量并行可以显著提升计算速度。

以下是 [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel) 模型在序列长度为 512 且不同批量大小情况下的单次前向推理的预期加速效果：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png">
</div>
