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

# Optimum Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a PyTorch quantization backend for [Optimum](https://huggingface.co/docs/optimum/index). It features linear quantization for weights (float8, int8, int4, int2) with accuracy very similar to full-precision models. Quanto is compatible with with any model modality and device, making it simple to use regardless of hardware.

Quanto is also compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for faster generation.

Install Quanto with the following command.

```bash
pip install optimum-quanto accelerate transformers
```

Quantize a model by creating a [`QuantoConfig`] and specifiying the `weights` parameter to quantize to. This works for any model in any modality as long as it contains [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) layers.

> [!TIP]
> The Transformers integration only supports weight quantization. Use the Quanto library directly if you need activation quantization, calibration, or QAT.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

quant_config = QuantoConfig(weights="int8")
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", 
    torch_dtype="auto", 
    device_map="auto", 
    quantization_config=quant_config
)
```

## torch.compile

Wrap a Quanto model with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for faster generation.

```py
import torch
from transformers import AutoModelForSpeechSeq2Seq, QuantoConfig

quant_config = QuantoConfig(weights="int8")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
  "openai/whisper-large-v2",
  torch_dtype="auto",
  device_map="auto",
  quantization_config=quant_config
)

model = torch.compile(model)
```

## Resources

Read the [Quanto: a PyTorch quantization backend for Optimum](https://huggingface.co/blog/quanto-introduction) blog post to learn more about the library design and benchmarks.

For more hands-on examples, take a look at the Quanto [notebook](https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing).