<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 양자화

양자화 기술은 더 적은 정보로 데이터를 표현하는 동시에 정확도를 너무 많이 잃지 않도록 하는 데 중점을 둡니다. 이는 종종 동일한 정보를 더 적은 비트로 표현하기 위해 데이터 유형을 변환하는 것을 의미합니다. 예를 들어, 모델 가중치가 32비트 부동 소수점으로 저장되어 있는데 이를 16비트 부동 소수점으로 양자화하면 모델 크기가 절반으로 줄어들어 저장하기 쉽고 메모리 사용량이 줄어듭니다. 또한 정밀도가 낮으면 더 적은 비트로 계산을 수행하는 데 시간이 덜 걸리므로 추론 속도가 빨라질 수 있습니다.

<Tip>

Transformers에 새로운 양자화 방법을 추가하고 싶으신가요? [HfQuantizer](./contribute) 가이드를 읽고 방법을 알아보세요!

</Tip>

<Tip>

양자화 분야를 처음 접하는 분이라면 DeepLearning.AI 양자화에 대한 초보자 친화적인 강좌를 확인해 보시기 바랍니다:

* [Hugging Face와 함께하는 양자화 기초](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [양자화 심화 과정](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

</Tip>

## 언제 무엇을 사용하나요?

개발자 커뮤니티에서는 다양한 사용 사례를 위해 많은 양자화 메소드를 개발해 왔습니다. 각 메소드에는 고유한 장단점이 있기 때문에 Transformers를 사용하면 사용 사례에 따라 이러한 통합된 메소드 중 하나를 실행할 수 있습니다.

예를 들어, 일부 양자화 메소드는 보다 정확하고 '극단적인' 압축(최대 1-2비트 양자화)을 위해 데이터 세트로 모델을 보정해야 하는 반면, 다른 메소드는 즉시 양자화하여 바로 사용할 수 있습니다.

고려해야 할 또 다른 매개변수는 대상 디바이스와의 호환성입니다. CPU, GPU 또는 Apple 실리콘에서 정량화하길 원하시나요?

요컨대, 다양한 양자화 방법을 지원하므로 원하는 목적에 가장 적합한 양자화 방법을 선택할 수 있습니다.

아래 표를 참조하여 어떤 양자화 메소드를 사용할지 결정하세요.

| Quantization method | On the fly quantization | CPU | CUDA GPU | RoCm GPU (AMD) | Metal (Apple Silicon) | torch.compile() support | Number of bits | Supports fine-tuning (through PEFT) | Serializable with 🤗 transformers | 🤗 transformers support | Link to library |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [AQLM](./aqlm) | 🔴 | 🟢 | 🟢 | 🔴 | 🔴 | 🟢 | 1 / 2 | 🟢 | 🟢 | 🟢 | https://github.com/Vahe1994/AQLM |
| [AWQ](./awq) | 🔴 | 🔴 | 🟢 | 🟢 | 🔴 | ? | 4 | 🟢 | 🟢 | 🟢 | https://github.com/casper-hansen/AutoAWQ |
| [bitsandbytes](./bitsandbytes) | 🟢 | 🟡 * | 🟢 | 🟡 * | 🔴 ** | 🔴    (soon!) | 4 / 8 | 🟢 | 🟢 | 🟢 | https://github.com/bitsandbytes-foundation/bitsandbytes |
| [compressed-tensors](./compressed_tensors) | 🔴 | 🟢 | 🟢 | 🟢 | 🔴 | 🔴 | 1 - 8 | 🟢 | 🟢 | 🟢 | https://github.com/neuralmagic/compressed-tensors |
| [EETQ](./eetq) | 🟢 | 🔴 | 🟢 | 🔴 | 🔴 | ? | 8 | 🟢 | 🟢 | 🟢 | https://github.com/NetEase-FuXi/EETQ |
| GGUF / GGML (llama.cpp) | 🟢 | 🟢 | 🟢 | 🔴 | 🟢 | 🔴 | 1 - 8 | 🔴 | [See GGUF section](../gguf) | [See GGUF section](../gguf) | https://github.com/ggerganov/llama.cpp |
| [GPTQ](./gptq) | 🔴 | 🔴 | 🟢 | 🟢 | 🔴 | 🔴 | 2 - 3 - 4 - 8 | 🟢 | 🟢 | 🟢 | https://github.com/AutoGPTQ/AutoGPTQ |
| [HQQ](./hqq) | 🟢 | 🟢 | 🟢 | 🔴 | 🔴 | 🟢 | 1 - 8 | 🟢 | 🔴 | 🟢 | https://github.com/mobiusml/hqq/ |
| [Quanto](./quanto) | 🟢 | 🟢 | 🟢 | 🔴 | 🟢 | 🟢 | 2 / 4 / 8 | 🔴 | 🔴 | 🟢 | https://github.com/huggingface/quanto |
| [FBGEMM_FP8](./fbgemm_fp8.md) | 🟢 | 🔴 | 🟢 | 🔴 | 🔴 | 🔴 | 8 | 🔴 | 🟢 | 🟢 | https://github.com/pytorch/FBGEMM |
| [torchao](./torchao.md) | 🟢 |  | 🟢 | 🔴 | partial support (int4 weight only) |  | 4 / 8 |  | 🟢🔴  | 🟢 | https://github.com/pytorch/ao |
<Tip>

\* bitsandbytes는 CUDA뿐 아니라 다른 여러 백엔드를 지원하기 위해 리팩토링 중입니다. 현재 ROCm(AMD GPU) 및 Intel CPU 구현이 완료되었으며, Intel XPU는 진행 중이고, Apple Silicon 지원은 4분기/내년 1분기 중으로 예정되어 있습니다. 설치 지침과 최신 백엔드 업데이트는 [이 링크](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend)를 참조하세요.

정식 출시 전에 버그를 파악하는 데 도움이 되는 여러분의 피드백을 소중하게 생각합니다! 자세한 내용과 피드백 링크는 [이 문서](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends)를 참조하세요.

</Tip>

<Tip>

\** bitsandbytes는 애플 실리콘 백엔드를 개발하고 이끌어갈 기여자를 찾고 있습니다. 관심이 있으신가요? 리포지토리를 통해 직접 문의하세요. 후원을 통해 급여를 받을 수도 있습니다.

</Tip>
