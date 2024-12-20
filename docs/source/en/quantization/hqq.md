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

# HQQ

[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq/) supports fast on-the-fly quantization for 8, 4, 3, 2, and even 1-bits. It doesn't require calibration data, and it is compatible with any model modality (LLMs, vision, etc.).

HQQ further supports fine-tuning with [PEFT](https://huggingface.co/docs/peft) and is fully compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster inference and training.

Install HQQ with the following command to get the latest version and to build its corresponding CUDA kernels.

```bash
pip install hqq
```

You can choose to either replace all the linear layers in a model with the same quantization config or dedicate a specific quantization config for specific linear layers.

<hfoptions id="hqq">
<hfoption id="replace all layers">

Quantize a model by creating a [`HqqConfig`] and specifying the `nbits` and `group_size` to replace for all the linear layers ([torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) of the model.

``` py
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

quant_config = HqqConfig(nbits=8, group_size=64)
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```

</hfoption>
<hfoption id="specific layers only">

Quantize a model by creating a dictionary specifying the `nbits` and `group_size` for the linear layers to quantize. Pass them to [`HqqConfig`] and set which layers to quantize with the config. This approach is especially useful for quantizing mixture-of-experts (MoEs) because they are less affected ly lower quantization settings.

``` py
q4_config = {'nbits':4, 'group_size':64}
q3_config = {'nbits':3, 'group_size':32}
quant_config  = HqqConfig(dynamic_config={
  'self_attn.q_proj':q4_config,
  'self_attn.k_proj':q4_config,
  'self_attn.v_proj':q4_config,
  'self_attn.o_proj':q4_config,

  'mlp.gate_proj':q3_config,
  'mlp.up_proj'  :q3_config,
  'mlp.down_proj':q3_config,
})

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```

</hfoption>
</hfoptions>

## Backends

HQQ supports various backends, including pure PyTorch and custom dequantization CUDA kernels. These backends are suitable for older GPUs and PEFT/QLoRA training.

```py
from hqq.core.quantize import *

HQQLinear.set_backend(HQQBackend.PYTORCH)
```

For faster inference, HQQ supports 4-bit fused kernels (torchao and Marlin) after a model is quantized. These can reach up to 200 tokens/sec on a single 4090. The example below demonstrates enabling the torchao_int4 backend.

```py
from hqq.utils.patching import prepare_for_inference

prepare_for_inference("model", backend="torchao_int4")
```

Refer to the [Backend](https://github.com/mobiusml/hqq/#backend) guide for more details.

## Resources

Read the [Half-Quadratic Quantization of Large Machine Learning Models](https://mobiusml.github.io/hqq_blog/) blog post for more details about HQQ.
