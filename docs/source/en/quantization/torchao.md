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

# TorchAO

[TorchAO](https://github.com/pytorch/ao) is an architecture optimization library for PyTorch, it provides high performance dtypes, optimization techniques and kernels for inference and training, featuring composability with native PyTorch features like `torch.compile`, FSDP etc.. Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks).

Before you begin, make sure the following libraries are installed with their latest version:

```bash
pip install --upgrade torch torchao
```

By default, the weights are loaded in full precision (torch.float32) regardless of the actual data type the weights are stored in such as torch.float16. Set `torch_dtype="auto"` to load the weights in the data type defined in a model's `config.json` file to automatically load the most memory-optimal data type.

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
# We support int4_weight_only, int8_weight_only and int8_dynamic_activation_int8_weight
# More examples and documentations for arguments can be found in https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques
quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# compile the quantized model to get speedup
import torchao
torchao.quantization.utils.recommended_inductor_config_setter()
quantized_model = torch.compile(quantized_model, mode="max-autotune")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# benchmark the performance
import torch.utils.benchmark as benchmark

def benchmark_fn(f, *args, **kwargs):
    # Manual warmup
    for _ in range(5):
        f(*args, **kwargs)
        
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"

MAX_NEW_TOKENS = 1000
print("int4wo-128 model:", benchmark_fn(quantized_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS))

bf16_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
bf16_model = torch.compile(bf16_model, mode="max-autotune")
print("bf16 model:", benchmark_fn(bf16_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS))

```

## Serialization and Deserialization
torchao quantization is implemented with [tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor), it only work with huggingface non-safetensor serialization and deserialization. It relies on `torch.load(..., weights_only=True)` to avoid arbitrary user code execution during load time and use [add_safe_globals](https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals) to allowlist some known user functions.

The reason why it does not support safe tensor serialization is that wrapper tensor subclass allows maximum flexibility so we want to make sure the effort of supporting new format of quantized Tensor is low, while safe tensor optimizes for maximum safety (no user code execution), it also means we have to make sure to manually support new quantization format.

```py
# save quantized model locally
output_dir = "llama3-8b-int4wo-128"
quantized_model.save_pretrained(output_dir, safe_serialization=False)

# push to huggingface hub
# save_to = "{user_id}/llama3-8b-int4wo-128"
# quantized_model.push_to_hub(save_to, safe_serialization=False)

# load quantized model
ckpt_id = "llama3-8b-int4wo-128"  # or huggingface hub model id
loaded_quantized_model = AutoModelForCausalLM.from_pretrained(ckpt_id, device_map="cuda")


# confirm the speedup
loaded_quantized_model = torch.compile(loaded_quantized_model, mode="max-autotune")
print("loaded int4wo-128 model:", benchmark_fn(loaded_quantized_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS))
```
