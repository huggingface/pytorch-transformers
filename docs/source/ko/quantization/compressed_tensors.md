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
# Compressed Tensors [[compressed-tensors]]

[`compressed-tensors`](https://github.com/neuralmagic/compressed-tensors) 라이브러리는 압축된 모델 체크포인트를 저장하고 관리할 수 있는 다재다능하고 효율적인 방법을 제공합니다. 이 라이브러리는 다양한 양자화 및 희소성 방식을 지원하여 GPTQ, AWQ, SmoothQuant, INT8, FP8, SparseGPT 등 다양한 모델 최적화를 처리할 수 있는 통합된 형식을 제공합니다.

지원되는 형식 중 일부는 다음과 같습니다:
1. `dense`
2. `int-quantized`: INT8 양자화 모델
    - 예시: [model/config](https://huggingface.co/nm-testing/tinyllama-w8a8-compressed-hf-quantizer)
3. `float-quantized`: FP8 양자화 모델; 현재 E4M3을 지원합니다
    - 예시: [model/config](https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-fp8-hf_compat/tree/main)
4. `pack-quantized`: INT32로 패킹된 형태의 INT4 또는 INT8 가중치 양자화 모델. INT4의 경우, 가중치는 INT4의 범위를 가지지만, 먼저 INT8로 저장된 후 INT32로 패킹됩니다.
    - 예시: [model/config](nm-testing/tinyllama-w4a16-compressed-hf-quantizer)

[llm-compressor](https://github.com/vllm-project/llm-compressor)을 사용하여 압축된 모델을 쉽게 생성할 수 있습니다.
또한, 모델은 독립적으로 생성된 후 compressed tensors 구성을 사용하여 직렬화할 수 있습니다.

Hugging Face Model Hub에서 기존 모델을 찾으려면, [`compressed-tensors` 태그](https://huggingface.co/models?other=compressed-tensors)를 검색하세요.

#### Features: [[features]]
 - 가중치 및 활성화 정밀도: FP8, INT4, INT8 (Q/DQ에서는 INT에 대해 임의의 정밀도가 허용됩니다)
 - 양자화 스케일 및 제로-포인트 전략: [tensor, channel, group, block, token](https://github.com/neuralmagic/compressed-tensors/blob/83b2e7a969d70606421a76b9a3d112646077c8de/src/compressed_tensors/quantization/quant_args.py#L43-L52)
 - 동적 토큰별 활성화 양자화 (또는 모든 정적 전략)
 - 희소성 적용 가능 
 - 선형 모듈뿐만 아니라 임의의 모듈 양자화 지원
 - 모듈 이름 또는 클래스로 특정 모듈 지원 또는 무시 가능

## Installation [[installation]]

compressed-tensors의 안정적인 배포를 [PyPI](https://pypi.org/project/compressed-tensors)에서 설치하는 것이 권장됩니다:
```bash
pip install compressed-tensors
```

최신 기능을 실험하고 싶은 개발자는 소스에서 패키지를 설치할 수도 있습니다:
```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

## Quickstart Model Load [[quickstart-model-load]]
양자화된 모델은 아래와 같이 간편하게 로드하여 추론에 사용할 수 있습니다. 현재는 이미 양자화된 모델만 불러올 수 있습니다. 모델을 compressed-tensors 형식으로 양자화 하려면 [llm-compressor](https://github.com/vllm-project/llm-compressor)를 참조하세요.

```python
from transformers import AutoModelForCausalLM

# compressed-tensors 형식으로 모델을 로드합니다
ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf")

# 메모리 사용량을 확인합니다
mem_params = sum([param.nelement()*param.element_size() for param in ct_model.parameters()])
print(f"{mem/2**30:.4f} GB")
# 8.4575 GB
```

위에서 확인할 수 있듯이 Llama 3.1 8B 모델의 compressed-tensors FP8 체크포인트는 양자화되지 않은 참조 체크포인트의 절반에 해당하는 메모리로 추론에 로드될 수 있습니다.

## Sample Use Cases - Load and run an FP8  [[compressed_tensors#sample-use-cases---load-and-run-an-fp8-model]]

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = [
    "Hello, my name is", #안녕하세요, 제 이름은
    "The capital of France is", #프랑스의 수도는
    "The future of AI is" #AI의 미래는
]

model_name = "nm-testing/Meta-Llama-3-8B-Instruct-fp8-hf_compat"

quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(prompt, return_tensors="pt")
generated_ids = quantized_model.generate(**inputs, max_length=50, do_sample=False)
outputs = tokenizer.batch_decode(generated_ids)

print(outputs)

"""
['<|begin_of_text|>Hello, my name is [Name]. I am a [Your Profession/Student] and I am here to learn about the [Course/Program] at [University/Institution]. I am excited to be here and I am looking forward to', '<|begin_of_text|>The capital of France is Paris, which is located in the north-central part of the country. Paris is the most populous city in France and is known for its stunning architecture, art museums, fashion, and romantic atmosphere. The city is home to', "<|begin_of_text|>The future of AI is here, and it's already changing the way we live and work. From virtual assistants to self-driving cars, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future of AI hold"] 
""" 
## ['<|begin_of_text|>안녕하세요, 제 이름은 [이름]입니다. 저는 [당신의 직업/학생]이며, [대학교/기관]에서 [과정/프로그램]에 대해 배우기 위해 여기에 왔습니다. 이곳에 오게 되어 매우 기쁘고, 앞으로 기대됩니다', '<|begin_of_text|>프랑스의 수도는 파리로, 이 나라는 북중부에 위치해 있습니다. 파리는 프랑스에서 가장 인구가 많은 도시이며, 멋진 건축물, 미술관, 패션, 그리고 로맨틱한 분위기로 유명합니다. 이 도시는', '<|begin_of_text|>AI의 미래는 이미 우리 곁에 와 있으며, 우리의 삶과 일하는 방식을 변화시키고 있습니다. 가상 비서에서 자율 주행차에 이르기까지, AI는 산업을 변혁시키고 우리가 기술과 상호작용하는 방식을 혁신하고 있습니다. 하지만 AI의 미래는 과연 무엇을 가져다줄까요]
]

```

위 예시는 `compressed-tensors`을 사용해서 생성 작업을 실행하는 간단한 예시입니다. 현재, 한 번 로드된 모델은 저장할 수 없습니다. 

## Deep dive into a compressed-tensors model checkpoint [[deep-dive-into-a-compressed-tensors-model-checkpoint]]

이 예시에서는 nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf compresseed-tensors 모델이 구성 항목을 통해 어떻게 정의되는지 살펴보고, 이것이 로드된 모델 표현으로 어떻게 변환되는지 확인해 보겠습니다.

먼저 [모델의 `quantization_config`](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf/blob/main/config.json)에 대해 알아보겠습니다. 처음 보면 항목이 많아 다소 복잡해 보일 수 있지만, 이는 compressed-tensors가 모델 압축 중 그리고 압축 후에도 유연하게 표현할 수 있는 형식이기 때문입니다.

실제로 체크포인트 로딩과 추론을 위해 설정을 단순화할 수 있으며, 기본값이나 비어있는 항목들을 포함하지 않도록 할 수 있습니다. 여기서는 압축이 실제로 어떻게 표현되는지에 초점을 맞추기 위해 설정을 간소화하겠습니다.

```yaml
"quantization_config": {
  "config_groups": {
    "group_0": {
      "input_activations": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      },
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      }
    }
  },
  "format": "naive-quantized",
  "ignore": ["lm_head"],
  "quant_method": "compressed-tensors",
  "quantization_status": "frozen"
},
```

위 구성에서 하나의 구성 그룹이 지정되어 있으며, 이는 FP8로 가중치 및 활성화 양자화를 수행하고, 정적인 텐서별 전략을 사용하는 것을 확인할 수 있습니다. 또한 `ignore` 리스트에 `lm_head` 모듈의 양자화를 건너뛰도록 항목이 포함되어 있어, 해당 모듈은 체크포인트에서 변경되지 않은 상태로 유지된다는 점도 주목해야합니다.

설정의 실제 결과를 확인하려면, [safetensors viewer](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf?show_file_info=model.safetensors.index.json)를 사용하여 모델 카드에서 모든 선형 모듈의 양자화된 가중치, 입력의 스케일, 그리고 가중치의 스케일을 모든 모델 레이어에서 확인할 수 있습니다. 

| Tensors | Shape |	Precision |
| ------- | ----- | --------- |
model.layers.0.input_layernorm.weight	| [4 096]	| BF16 
model.layers.0.mlp.down_proj.input_scale	| [1]	| BF16 
model.layers.0.mlp.down_proj.weight	| [4 096, 14 336] |	F8_E4M3 
model.layers.0.mlp.down_proj.weight_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.input_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.weight	| [14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.gate_proj.weight_scale	| [1] |	BF16 
model.layers.0.mlp.up_proj.input_scale|	[1]	|BF16 
model.layers.0.mlp.up_proj.weight |	[14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.up_proj.weight_scale | [1]	| BF16 
model.layers.0.post_attention_layernorm.weight |	[4 096]	|BF16 
model.layers.0.self_attn.k_proj.input_scale |	[1]	|  BF16
model.layers.0.self_attn.k_proj.weight |	[1 024, 4 096]|	F8_E4M3
model.layers.0.self_attn.k_proj.weight_scale |[1]	| BF16 
model.layers.0.self_attn.o_proj.input_scale	| [1]	| BF16
model.layers.0.self_attn.o_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.o_proj.weight_scale | [1]	| BF16 
model.layers.0.self_attn.q_proj.input_scale	| [1]	| BF16 
model.layers.0.self_attn.q_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.q_proj.weight_scale |	[1] | BF16 
model.layers.0.self_attn.v_proj.input_scale	| [1] | BF16 
model.layers.0.self_attn.v_proj.weight |	[1 024, 4 096]	| F8_E4M3 
model.layers.0.self_attn.v_proj.weight_scale |	[1] |	BF16 

compressed-tensors HFQuantizer 통합을 사용하여 모델을 로드하면, 양자화 설정 내에서 지정된 모든 선형 모듈이 `CompressedLinear` 모듈로 대체된 것을 확인할 수 있습니다. 해당 모듈은 압축된 가중치와 추론을 위한 순전파를 처리합니다. 이전에 ignore 리스트에 언급된 `lm_head`는 여전히 양자화되지 않은 선형 모듈로 유지된다는 점을 주목할 필요가 있습니다.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf")
print(ct_model)
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (k_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (v_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (o_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (up_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (down_proj): CompressedLinear(
            in_features=14336, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""
```
