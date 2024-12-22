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

# VPTQ 

> [!TIP]
> Try VPTQ on [Hugging Face](https://huggingface.co/spaces/microsoft/VPTQ)!
> Try VPTQ on [Google Colab](https://colab.research.google.com/github/microsoft/VPTQ/blob/main/notebooks/vptq_example.ipynb)!
> Know more about VPTQ on [ArXiv](https://arxiv.org/pdf/2409.17066)!

Vector Post-Training Quantization ([VPTQ](https://github.com/microsoft/VPTQ)) is a novel Post-Training Quantization method that leverages Vector Quantization to high accuracy on LLMs at an extremely low bit-width (<2-bit). VPTQ can compress 70B, even the 405B model, to 1-2 bits without retraining and maintain high accuracy.

- Better Accuracy on 1-2 bits, (405B @ <2bit, 70B @ 2bit)
- Lightweight Quantization Algorithm: only cost ~17 hours to quantize 405B Llama-3.1
- Agile Quantization Inference: low decode overhead, best throughput, and TTFT

Inference support for VPTQ is released in the `vptq` library. Make sure to install it to run the models:
```bash
pip install vptq
```

The library provides efficient kernels for NVIDIA/AMD GPU inference.

To run VPTQ models simply load a model that has been quantized with VPTQ:

## Inference example
**Run Llama 3.1 70b on RTX4090 (24G @ ~2bits) in real time**
![Llama3 1-70b-prompt](https://github.com/user-attachments/assets/d8729aca-4e1d-4fe1-ac71-c14da4bdd97f)


```python
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft",
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft")
input_ids = tokenizer("hello, it's me", return_tensors="pt").to("cuda")
out = model.generate(**input_ids, max_new_tokens=32, do_sample=False)
```

## Quantize your own model
VPTQ algorithm early-released at [VPTQ ](https://github.com/microsoft/VPTQ/tree/algorithm), 
and checkout the [tutorial](https://github.com/microsoft/VPTQ/blob/algorithm/algorithm.md).

## Early Results from Tech Report
VPTQ achieves better accuracy and higher throughput with lower quantization overhead across models of different sizes. The following experimental results are for reference only; VPTQ can achieve better outcomes under reasonable parameters, especially in terms of model accuracy and inference speed.


| Model       | bitwidth | W2↓  | C4↓  | AvgQA↑ | tok/s↑ | mem(GB) | cost/h↓ |
| ----------- | -------- | ---- | ---- | ------ | ------ | ------- | ------- |
| LLaMA-2 7B  | 2.02     | 6.13 | 8.07 | 58.2   | 39.9   | 2.28    | 2       |
|             | 2.26     | 5.95 | 7.87 | 59.4   | 35.7   | 2.48    | 3.1     |
| LLaMA-2 13B | 2.02     | 5.32 | 7.15 | 62.4   | 26.9   | 4.03    | 3.2     |
|             | 2.18     | 5.28 | 7.04 | 63.1   | 18.5   | 4.31    | 3.6     |
| LLaMA-2 70B | 2.07     | 3.93 | 5.72 | 68.6   | 9.7    | 19.54   | 19      |
|             | 2.11     | 3.92 | 5.71 | 68.7   | 9.7    | 20.01   | 19      |



## More Models in [VPTQ-community](https://huggingface.co/VPTQ-community) 

⚠️ The repository only provides a method of model quantization algorithm. 

⚠️ The open-source community VPTQ-community provides models based on the technical report and quantization algorithm.



**Quick Estimation of Model Bitwidth (Excluding Codebook Overhead)**:

- **Model Naming Convention**: The model's name includes the **vector length** $v$, **codebook (lookup table) size**, and **residual codebook size**. For example, "Meta-Llama-3.1-70B-Instruct-v8-k65536-256-woft" is "Meta-Llama-3.1-70B-Instruct", where:
  - **Vector Length**: 8
  - **Number of Centroids**: 65536 (2^16)
  - **Number of Residual Centroids**: 256 (2^8)
- **Equivalent Bitwidth Calculation**:
  - **Index**: log2(65536) = 16 / 8 = 2 bits
  - **Residual Index**: log2(256) = 8 / 8 = 1 bit
  - **Total Bitwidth**: 2 + 1 = 3 bits
- **Model Size Estimation**: 70B * 3 bits / 8 bits per Byte = 26.25 GB

- **Note**: This estimate does not include the size of the codebook (lookup table), other parameter overheads, and the padding overhead for storing indices. For the detailed calculation method, please refer to **Tech Report Appendix C.2**.


|            Model Series            |  Collections                              | (Estimated) Bit per weight     |
| :--------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|       Llama 3.1 Nemotron 70B Instruct HF        |  [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-llama-31-nemotron-70b-instruct-hf-without-finetune-671730b96f16208d0b3fe942)  | [4 bits](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v8-k65536-256-woft) [2 bits (1)](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v16-k65536-65536-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v8-k65536-0-woft) [1.875 bits](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v16-k65536-16384-woft) [1.625 bits](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v16-k65536-1024-woft) [1.5 bits](https://huggingface.co/VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v16-k65536-256-woft) |
|       Llama 3.1 8B Instruct        |  [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-llama-31-8b-instruct-without-finetune-66f2b70b1d002ceedef02d2e)  | [4 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-65536-woft) [3.5 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-4096-woft) [3 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-256-woft) [2.3 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft)                                                                                                                                                                                                                                                                                                                                                                                                              |
|       Llama 3.1 70B Instruct       | [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-llama-31-70b-instruct-without-finetune-66f2bf454d3dd78dfee2ff11)  | [4 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-256-woft) [2.25 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-4-woft)  [2 bits (1)](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft) [1.93 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-32768-woft) [1.875 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k32768-0-woft) [1.75 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k16384-0-woft) |
|      Llama 3.1 405B Instruct       | [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-llama-31-405b-instruct-without-finetune-66f4413f9ba55e1a9e52cfb0) | [4 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v8-k65536-256-woft) [2 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-65536-woft) [1.875 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k32768-32768-woft) [1.625 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-1024-woft) [1.5 bits (1)](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v8-k4096-0-woft) [1.5 bits (2)](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-256-woft) [1.43 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-128-woft) [1.375 bits](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-64-woft) |
| Mistral Large Instruct 2407 (123B) | [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-mistral-large-instruct-2407-without-finetune-6711ebfb7faf85eed9cceb16) | [4 bits](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v8-k65536-256-woft) [2 bits (1)](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v16-k65536-65536-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v8-k65536-0-woft) [1.875 bits](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v16-k65536-16384-woft) [1.75 bits](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v16-k65536-4096-woft) [1.625 bits](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v16-k65536-1024-woft) [1.5 bits](https://huggingface.co/VPTQ-community/Mistral-Large-Instruct-2407-v16-k65536-256-woft) |
|        Qwen 2.5 7B Instruct        |  [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-qwen-25-7b-instruct-without-finetune-66f3e9866d3167cc05ce954a)   | [4 bits](https://huggingface.co/VPTQ-community/Qwen2.5-7B-Instruct-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Qwen2.5-7B-Instruct-v8-k65536-256-woft) [2 bits (1)](https://huggingface.co/VPTQ-community/Qwen2.5-7B-Instruct-v8-k256-256-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Qwen2.5-7B-Instruct-v8-k65536-0-woft)  [2 bits (3)](https://huggingface.co/VPTQ-community/Qwen2.5-7B-Instruct-v16-k65536-65536-woft)                                                                                                                                                                                                                                                                                                                                              |
|       Qwen 2.5 14B Instruct        |  [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-qwen-25-14b-instruct-without-finetune-66f827f83c7ffa7931b8376c)  | [4 bits](https://huggingface.co/VPTQ-community/Qwen2.5-14B-Instruct-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Qwen2.5-14B-Instruct-v8-k65536-256-woft) [2 bits (1)](https://huggingface.co/VPTQ-community/Qwen2.5-14B-Instruct-v8-k256-256-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Qwen2.5-14B-Instruct-v8-k65536-0-woft)  [2 bits (3)](https://huggingface.co/VPTQ-community/Qwen2.5-14B-Instruct-v16-k65536-65536-woft)                                                                                                                                                                                                                                                                                                                                         |
|       Qwen 2.5 32B Instruct        |  [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-qwen-25-32b-instruct-without-finetune-66fe77173bf7d64139f0f613)  | [4 bits](https://huggingface.co/VPTQ-community/Qwen2.5-32B-Instruct-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Qwen2.5-32B-Instruct-v8-k65536-256-woft) [2 bits (1)](https://huggingface.co/VPTQ-community/Qwen2.5-32B-Instruct-v16-k65536-65536-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Qwen2.5-32B-Instruct-v8-k65536-0-woft) [2 bits (3)](https://huggingface.co/VPTQ-community/Qwen2.5-32B-Instruct-v8-k256-256-woft)                                                                                                                                                                                                                                                                                                                                          |
|       Qwen 2.5 72B Instruct        |  [HF 🤗](https://huggingface.co/collections/VPTQ-community/vptq-qwen-25-72b-instruct-without-finetune-66f3bf1b3757dfa1ecb481c0)  | [4 bits](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v8-k65536-65536-woft) [3 bits](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v8-k65536-256-woft) [2.38 bits](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v8-k1024-512-woft) [2.25 bits (1)](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v8-k512-512-woft) [2.25 bits (2)](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v8-k65536-4-woft) [2 bits (1)](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v8-k65536-0-woft) [2 bits (2)](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v16-k65536-65536-woft) [1.94 bits](https://huggingface.co/VPTQ-community/Qwen2.5-72B-Instruct-v16-k65536-32768-woft)                                                  |
|  Reproduced from the tech report   |     [HF 🤗](https://huggingface.co/collections/VPTQ-community/reproduced-vptq-tech-report-baseline-66fbf1dffe741cc9e93ecf04)     | Results from the open source community for reference only, please use them responsibly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Hessian and Inverse Hessian Matrix |      [HF 🤗](https://huggingface.co/collections/VPTQ-community/hessian-and-invhessian-checkpoints-66fd249a104850d17b23fd8b)      | Collected from RedPajama-Data-1T-Sample, following [Quip#](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/hessian_offline_llama.py)                                                                                                                                                                                                               