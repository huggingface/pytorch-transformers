<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Multi-GPU inference

Built-in Tensor Parallelism (TP) is now available with certain models using PyTorch. Tensor parallelism shards a model onto multiple GPUs, enabling larger model sizes, and parallelizes computations such as matrix multiplication.

To enable tensor parallel, pass the argument `tp_plan="auto"` to [`~AutoModelForCausalLM.from_pretrained`]:

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize distributed
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

# Retrieve tensor parallel model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",
)

# Prepare input tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Distributed run
outputs = model(inputs)
```

You can use `torchrun` to launch the above script with multiple processes, each mapping to a GPU:

```
torchrun --nproc-per-node 4 demo.py
```

PyTorch tensor parallel is currently supported for the following models:
* [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)

You can request to add tensor parallel support for another model by opening a GitHub Issue or Pull Request.

### Expected speedups

You can benefit from considerable speedups for inference, especially for inputs with large batch size or long sequences.

For a single forward pass on [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel) with a sequence length of 512 and various batch sizes, the expected speedup is as follows:

<div style="text-align: center">
<img src="huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct, seqlen = 512, python, w_ compile.png">
</div>
