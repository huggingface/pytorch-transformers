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

# DiffLlama

## Overview

The DiffLlama model was proposed in [Differential Transformer](https://arxiv.org/abs/2410.05258) by Kazuma Matsumoto and .
This model is combine Llama model and Differential Transformer's Attention.

The abstract from the paper is the following:

*Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, the differential attention mechanism calculates attention scores as the difference between two separate softmax attention maps. The subtraction cancels noise, promoting the emergence of sparse attention patterns. Experimental results on language modeling show that Diff Transformer outperforms Transformer in various settings of scaling up model size and training tokens. More intriguingly, it offers notable advantages in practical applications, such as long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and reduction of activation outliers. By being less distracted by irrelevant context, Diff Transformer can mitigate hallucination in question answering and text summarization. For in-context learning, Diff Transformer not only enhances accuracy but is also more robust to order permutation, which was considered as a chronic robustness issue. The results position Diff Transformer as a highly effective and promising architecture to advance large language models.*

### Usage tips
The hyperparameters of this model is the same as Llama model.


## DiffLlamaConfig

[[autodoc]] DiffLlamaConfig

## DiffLlamaModel

[[autodoc]] DiffLlamaModel
    - forward

## DiffLlamaForCausalLM

[[autodoc]] DiffLlamaForCausalLM
    - forward

## DiffLlamaForSequenceClassification

[[autodoc]] DiffLlamaForSequenceClassification
    - forward

## DiffLlamaForQuestionAnswering

[[autodoc]] DiffLlamaForQuestionAnswering
    - forward

## DiffLlamaForTokenClassification

[[autodoc]] DiffLlamaForTokenClassification
    - forward
