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

# FBGEMM FP8

With FBGEMM FP8 quantization method, you can quantize your model in FP8 (W8A8):
- the weights will be quantized in 8bit (FP8) per channel
- the activation will be quantized in 8bit (FP8) per token

It relies on the [FBGEMM](https://github.com/pytorch/FBGEMM) library which provides efficient low-precision general matrix multiplication for small batch sizes and support for accuracy-loss minimizing techniques such as row-wise quantization and outlier-aware quantization. 

> [!TIP]
> You need a GPU with compute capability>=9 (e.g. H100) 

Before you begin, make sure the following libraries are installed with their latest version:

```bash
pip install --upgrade accelerate fbgemm-gpu torch
```

If you are having issues with fbgemm-gpu and torch library, you might need to install the nightly release. You can follow the instruction [here](https://pytorch.org/FBGEMM/fbgemm_gpu-development/InstallationInstructions.html#fbgemm-gpu-install-libraries:~:text=found%20here.-,Install%20the%20FBGEMM_GPU%20Package,-Install%20through%20PyTorch)

By default, the weights are loaded in full precision (torch.float32) regardless of the actual data type the weights are stored in such as torch.float16. Set `torch_dtype="auto"` to load the weights in the data type defined in a model's `config.json` file to automatically load the most memory-optimal data type.

```py
from transformers import FbgemmFp8Config, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = FbgemmFp8Config()
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

A quantized model can be saved via "saved_pretrained" and be reused again via the "from_pretrained".

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```