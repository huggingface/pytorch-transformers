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


# xLSTM

## Overview

The xLSTM model was proposed in [xLSTM: Extended Long Short-Term Memory](https://openreview.net/forum?id=ARAxPPIAhq) by Maximilian Beck*, Korbinian Pöppel*, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter and Sepp Hochreiter.
xLSTM updates the original LSTM architecture to be competitive with Transformer models by introducing exponential gating, matrix memory expansion, and parallelizable training and ingestion.

The [7B model](https://hf.co/NX-AI/xLSTM-7b) variant was trained by the xLSTM team Maximilian Beck, Korbinian Pöppel, Phillip Lippe, Richard Kurle, Patrick Blies, Sebastian Böck and Sepp Hochreiter at NXAI.

The abstract from the paper is the following:

*In the 1990s, the constant error carousel and gating were introduced as the central ideas of the Long Short-Term Memory (LSTM). Since then, LSTMs have stood the test of time and contributed to numerous deep learning success stories, in particular they constituted the first Large Language Models (LLMs). However, the advent of the Transformer technology with parallelizable self-attention at its core marked the dawn of a new era, outpacing LSTMs at scale. We now raise a simple question: How far do we get in language modeling when scaling LSTMs to billions of parameters, leveraging the latest techniques from modern LLMs, but mitigating known limitations of LSTMs? Firstly, we introduce exponential gating with appropriate normalization and stabilization techniques. Secondly, we modify the LSTM memory structure, obtaining: (i) sLSTM with a scalar memory, a scalar update, and new memory mixing, (ii) mLSTM that is fully parallelizable with a matrix memory and a covariance update rule. Integrating these LSTM extensions into residual block backbones yields xLSTM blocks that are then residually stacked into xLSTM architectures. Exponential gating and modified memory structures boost xLSTM capabilities to perform favorably when compared to state-of-the-art Transformers and State Space Models, both in performance and scaling.*

This model was contributed by [NX-AI](https://huggingface.co/NX-AI).
The original code can be found [here](https://github.com/NX-AI/xlstm).


## xLSTMConfig

[[autodoc]] xLSTMConfig

## xLSTMModel

[[autodoc]] xLSTMModel
    - forward

## xLSTMLMHeadModel

[[autodoc]] xLSTMForCausalLM
    - forward
