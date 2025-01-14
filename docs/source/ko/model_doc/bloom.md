<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BLOOM [[bloom]]

## 개요 [[overview]]

BLOOM 모델은 [BigScience Workshop](https://bigscience.huggingface.co/)를 통해 다양한 버전으로 제안되었습니다. BigScience는 연구자들이 시간을 모으고 자원을 활용하여 집단적으로 더 큰 영향을 이루는 다른 오픈 사이언스 이니셔티브에서 영감을 받았습니다. BLOOM의 아키텍처는 본질적으로 GPT-3(다음 토큰 예측을 위한 자동 회귀 모델)와 유사하지만, 46개 언어와 13개 프로그래밍 언어로 훈련되었다는 점에서 차이가 있습니다. 동일한 데이터셋으로 여러 개의 작은 버전 모델들을 훈련했습니다. BLOOM은 다음과 같은 버전으로 제공됩니다:

- [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
- [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
- [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bloom](https://huggingface.co/bigscience/bloom) (176B parameters)

## 리소스 [[resources]]

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BLOOM. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

BLOOM을 시작하는 데 도움이 될 공식 Hugging Face 및 커뮤니티(🌎로 표시된) 리소스 목록입니다. 추가로 관련 리소스를 제출하고 싶다면 언제든지 Pull Request를 열어주시면, 검토 후 추가하겠습니다! 리소스를 제출해주실 때에는 기존 것과 중복되지 않는 새로운 리소스를 보내주세요!

<PipelineTag pipeline="text-generation"/>

- [`BloomForCausalLM`] 는 [일상 언어 모델링 예시 스크립트 causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 의 지원을 받았습니다.

기타 참고할 만한 자료:
- [일상 언어 모델링 작업 가이드 Causal language modeling task guide](../tasks/language_modeling)
- [텍스트 분류 작업 가이드 Text classification task guide](../tasks/sequence_classification)
- [토큰 분류 작업 가이드 Token classification task guide](../tasks/token_classification)
- [질의 응답 작업 가이드 Question answering task guide](../tasks/question_answering)


⚡️ 추론
- [최적화 스토리 : Bloom 추론 (Optimization story: Bloom inference)](https://huggingface.co/blog/bloom-inference-optimization)에 대한 블로그글
- [DeepSpeed와 Accelerate를 사용하여 놀랍도록 빠른 Bloom 추론 (Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate)](https://huggingface.co/blog/bloom-inference-pytorch-scripts) 에 대한 블로그글

⚙️ 학습
- [Bloom 학습 뒤 기술 (The Technology Behind BLOOM Training)](https://huggingface.co/blog/bloom-megatron-deepspeed) 에 대한 블로그글

## BloomConfig [[transformers.BloomConfig]] 

[[autodoc]] BloomConfig
    - all

## BloomTokenizerFast [[transformers.BloomTokenizerFast]]

[[autodoc]] BloomTokenizerFast
    - all


<frameworkcontent>
<pt>

## BloomModel [[transformers.BloomModel]]

[[autodoc]] BloomModel
    - forward

## BloomForCausalLM [[transformers.BloomForCausalLM]]

[[autodoc]] BloomForCausalLM
    - forward

## BloomForSequenceClassification [[transformers.BloomForSequenceClassification]]

[[autodoc]] BloomForSequenceClassification
    - forward

## BloomForTokenClassification [[transformers.BloomForTokenClassification]]

[[autodoc]] BloomForTokenClassification
    - forward

## BloomForQuestionAnswering [[transformers.BloomForQuestionAnswering]]

[[autodoc]] BloomForQuestionAnswering
    - forward

</pt>
<jax>

## FlaxBloomModel [[transformers.FlaxBloomModel]]

[[autodoc]] FlaxBloomModel
    - __call__

## FlaxBloomForCausalLM [[transformers.FlaxBloomForCausalLM]]

[[autodoc]] FlaxBloomForCausalLM
    - __call__

</jax>
</frameworkcontent>


