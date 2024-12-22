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

# 양자화[[quantization]]



양자화 기법은 가중치와 활성화를 8비트 정수(int8)와 같은 더 낮은 정밀도의 데이터 타입으로 표현함으로써 메모리와 계산 비용을 줄입니다. 이를 통해 일반적으로는 메모리에 올릴 수 없는 더 큰 모델을 로드할 수 있고, 추론 속도를 높일 수 있습니다. Transformers는 AWQ와 GPTQ 양자화 알고리즘을 지원하며, bitsandbytes를 통해 8비트와 4비트 양자화를 지원합니다.
Transformers에서 지원되지 않는 양자화 기법들은 [`HfQuantizer`] 클래스를 통해 추가될 수 있습니다.

<Tip>

모델을 양자화하는 방법은 이 [양자화](../quantization) 가이드를 통해 배울 수 있습니다.

</Tip>

## QuantoConfig[[transformers.QuantoConfig]]

[[autodoc]] QuantoConfig

## AqlmConfig[[transformers.AqlmConfig]]

[[autodoc]] AqlmConfig

## VptqConfig[[transformers.VptqConfig]]

[[autodoc]] VptqConfig

## AwqConfig[[transformers.AwqConfig]]

[[autodoc]] AwqConfig

## EetqConfig[[transformers.EetqConfig]]
[[autodoc]] EetqConfig

## GPTQConfig[[transformers.GPTQConfig]]

[[autodoc]] GPTQConfig

## BitsAndBytesConfig[[#transformers.BitsAndBytesConfig]]

[[autodoc]] BitsAndBytesConfig

## HfQuantizer[[transformers.quantizers.HfQuantizer]]

[[autodoc]] quantizers.base.HfQuantizer

## HqqConfig[[transformers.HqqConfig]]

[[autodoc]] HqqConfig

## FbgemmFp8Config[[transformers.FbgemmFp8Config]]

[[autodoc]] FbgemmFp8Config

## CompressedTensorsConfig[[transformers.CompressedTensorsConfig]]

[[autodoc]] CompressedTensorsConfig

## TorchAoConfig[[transformers.TorchAoConfig]]

[[autodoc]] TorchAoConfig
