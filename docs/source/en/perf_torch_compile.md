<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# torch.compile

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) compiles PyTorch code into optimized kernels that significantly speed up inference. This feature relies on TorchDynamo to compile the code into graphs and TorchInductor to further compile the graphs into optimized kernels. It is a powerful optimization tool and in many cases, it only requires adding a single line of code.

Wrap a model with torch.compile to compile and return an optimized model.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
compiled_model = torch.compile(model)
```

> [!TIP]
> The initial call to torch.compile is slow because the model needs to be compiled. Subsequent calls to the compiled model are much faster because it doesn't need to compile again.

There are several parameters you can use to customize the compilation process. Two of the more important ones are listed below. For a full list of parameters, refer to the [torch.compile documentation](https://pytorch.org/docs/stable/generated/torch.compile.html).

## Modes

The `mode` parameter offers several performance options for compiling, and you should try different modes to see which one works best for your use case.

- `default` is a balanced option between speed and memory.
- `reduce-overhead` reduces the Python overhead at the expense of a little more memory, but it can be faster.
- `max-autotune` offers the fastest speed, but compiling the code takes longer.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
compiled_model = torch.compile(model, mode="reduce-overhead")
```

## Fullgraph

Fullgraph attempts to compile the entire model into a single graph to maximize performance. torch.compile raises an error if it encounters a graph break, which means it can't compile the model into a single graph.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

## Benchmark results

Refer to the table below for performance benchmarks comparing the mean inference time in milliseconds with torch.compile enabled and disabled across various GPUs and batch sizes on the same image. Select **Subset** in the table below to switch between different GPUs, as well as benchmarks on [PyTorch nightly 2.1.0dev](https://download.pytorch.org/whl/nightly/cu118) and torch.compile with `reduce-overhead` mode enabled.

<iframe
  src="https://huggingface.co/datasets/stevhliu/compile-benchmarks/embed/viewer/t4/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
