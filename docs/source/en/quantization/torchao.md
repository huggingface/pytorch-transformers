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

# torchao

[torchao](https://github.com/pytorch/ao) is a PyTorch architecture optimization library with support for custom high performance data types, quantization, and sparsity. It is composable with native PyTorch features such as [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster inference and training.

Install torchao with the following command.

```bash
# Updating 🤗 Transformers to the latest version, as the example script below uses the new auto compilation
pip install --upgrade torch torchao transformers
```

torchao supports int8 weight quantization and int8 dynamic quantization of weights. Create a [`TorchAoConfig`] and specify the quantization type and `group_size` of the weights to quantize.

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

## torch.compile

Wrap the quantized model with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster generation.

```py
import torchao

quantized_model = torch.compile(quantized_model, mode="max-autotune")
```

## Serialization

torchao implements [torch.Tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) for maximum flexibility in supporting new quantized torch.Tensor formats. [Safetensors](https://huggingface.co/docs/safetensors/en/index) serialization and deserialization does not work with torchaco.

To avoid arbitrary user code execution, torchao sets `weights_only=True` in [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html) to ensure only tensors are loaded. Any known user functions can be whitelisted with [add_safe_globals](https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals).

```py
# don't serialize model with Safetensors
output_dir = "llama3-8b-int4wo-128"
quantized_model.save_pretrained("llama3-8b-int4wo-128", safe_serialization=False)
```

## Resources

For a better sense of expected performance, view the [benchmarks](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks) for various models with CUDA and XPU backends.

Refer to [Other Available Quantization Techniques](https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques) for more examples and documentation.
