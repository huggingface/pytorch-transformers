<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSpeed [[deepspeed]]

[DeepSpeed](https://github.com/microsoft/DeepSpeed)는 Zero Redundancy Optimizer (ZeRO)를 기반으로 매우 큰 모델을 GPU에 맞춰 학습시키기 위한 최적화 라이브러리입니다. ZeRO는 여러 단계로 제공되며, 각 단계는 옵티마이저 상태, 기울기, 파라미터를 분할하고 CPU 또는 NVMe로 오프로드할 수 있게 하여, 점진적으로는 더 많은 GPU 메모리를 절약합니다. DeepSpeed는 [`Trainer`] 클래스와 통합되어 있으며, 대부분의 설정이 자동으로 처리됩니다.

[`Trainer`] 없이 DeepSpeed를 사용하고 싶다면, Transformers에서 제공하는 [`HfDeepSpeedConfig`] 클래스를 사용할 수 있습니다.

<Tip>

[DeepSpeed](../deepspeed)가이드에 있는 [`Trainer`]를 사용해서 DeepSpeed에 대해 더 배워보세요.

</Tip>

## HfDeepSpeedConfig [[transformers.integrations.HfDeepSpeedConfig]]

[[autodoc]] integrations.HfDeepSpeedConfig
    - all
