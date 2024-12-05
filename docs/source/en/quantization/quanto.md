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

# Optimum-quanto

<Tip>

Try optimum-quanto + transformers with this [notebook](https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing)!

</Tip>


[🤗 optimum-quanto](https://github.com/huggingface/optimum-quanto) library is a versatile pytorch quantization toolkit. The quantization method used is the linear quantization. Quanto provides several unique features such as:

- weights quantization (`float8`,`int8`,`int4`,`int2`)
- activation quantization (`float8`,`int8`)
- modality agnostic (e.g CV,LLM)
- device agnostic (e.g CUDA,XPU,MPS,CPU)
- compatibility with `torch.compile`
- easy to add custom kernel for specific device
- supports quantization aware training
<!-- Add link to the blogpost -->

Before you begin, make sure the following libraries are installed:

```bash
pip install optimum-quanto accelerate transformers
```

Now you can quantize a model by passing [`QuantoConfig`] object in the [`~PreTrainedModel.from_pretrained`] method. This works for any model in any modality, as long as it contains `torch.nn.Linear` layers. 

The integration with transformers only supports weights quantization. For the more complex use case such as activation quantization, calibration and quantization aware training, you should use [optimum-quanto](https://github.com/huggingface/optimum-quanto) library instead.

By default, the weights are loaded in full precision (torch.float32) regardless of the actual data type the weights are stored in such as torch.float16. Set `torch_dtype="auto"` to load the weights in the data type defined in a model's `config.json` file to automatically load the most memory-optimal data type.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = QuantoConfig(weights="int8")
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cuda:0", quantization_config=quantization_config)
```

Note that serialization is not supported yet with transformers but it is coming soon! If you want to save the model, you can use quanto library instead.

Optimum-quanto library uses linear quantization algorithm for quantization. Even though this is a basic quantization technique, we get very good results! Have a look at the following benchmark (llama-2-7b on perplexity metric). You can find more benchmarks [here](https://github.com/huggingface/optimum-quanto/tree/main/bench/generation)

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/NousResearch-Llama-2-7b-hf_Perplexity.png" alt="llama-2-7b-quanto-perplexity" />
  </div>
</div>

The library is versatile enough to be compatible with most PTQ optimization algorithms. The plan in the future is to integrate the most popular algorithms in the most seamless possible way (AWQ, Smoothquant).