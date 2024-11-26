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

# OLMo2

## Overview

The OLMo2 model is the successor of the OLMo model, which was proposed in
[OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838).

 The architectural changes from the original OLMo model to this model are:

- RMSNorm is used instead of standard layer norm.
- Norm is applied to attention queries and keys.
- Norm is applied after attention/feedforward layers rather than before.

This model was contributed by [shanearora](https://huggingface.co/shanearora).
The original code can be found [here](https://github.com/allenai/OLMo/tree/main/olmo).


## Olmo2Config

[[autodoc]] Olmo2Config

## Olmo2Model

[[autodoc]] Olmo2Model
    - forward

## Olmo2ForCausalLM

[[autodoc]] Olmo2ForCausalLM
    - forward
