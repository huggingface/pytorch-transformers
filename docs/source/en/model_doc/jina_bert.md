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

# JinaBERT

## Overview

The JinaBERT model was proposed in [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923) by Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, Han Xiao.
It is an multilingual embedding model speaks English and 30 widely used programming languages.
It is based on a Bert architecture that supports the symmetric bidirectional variant of [ALiBi](https://arxiv.org/abs/2108.12409) to allow longer sequence length.
The abstract from the paper is the following:

*Text embedding models have emerged as powerful tools for transforming sentences into fixed-sized feature vectors that encapsulate semantic information.
 While these models are essential for tasks like information retrieval,
 semantic clustering, and text re-ranking, most existing open-source models, especially those built on architectures like BERT,
 struggle to represent lengthy documents and often resort to truncation.
 One common approach to mitigate this challenge involves splitting documents into smaller paragraphs for embedding.
 However, this strategy results in a much larger set of vectors, consequently leading to increased memory consumption and computationally intensive vector searches with elevated latency.*

*To address these challenges, we introduce Jina Embeddings 2, an open-source text embedding model capable of accommodating up to 8192 tokens.
 This model is designed to transcend the conventional 512-token limit and adeptly process long documents.
 Jina Embeddings 2 not only achieves state-of-the-art performance on a range of embedding-related tasks in the MTEB benchmark but also matches the performance of OpenAI's proprietary ada-002 model.
 Additionally, our experiments indicate that an extended context can enhance performance in tasks such as NarrativeQA.*

This model was contributed by [joelkoch](https://huggingface.co/joelkoch).
The original code can be found [here](https://huggingface.co/jinaai/jina-bert-v2-qk-post-norm/tree/main).

## Supported (Programming) Languages
- English
- Assembly
- Batchfile
- C
- C#
- C++
- CMake
- CSS
- Dockerfile
- FORTRAN
- GO
- Haskell
- HTML
- Java
- JavaScript
- Julia
- Lua
- Makefile
- Markdown
- PHP
- Perl
- PowerShell
- Python
- Ruby
- Rust
- SQL
- Scala
- Shell
- TypeScript
- TeX
- Visual Basic

## Usage tips

Please apply mean pooling when integrating the model.

### Why mean pooling?

`mean poooling` takes all token embeddings from model output and averaging them at sentence/paragraph level.
It has been proved to be the most effective way to produce high-quality sentence embeddings.
We offer an `encode` function to deal with this.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with JinaBERT. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- [Jina AI Discord community](https://discord.jina.ai) 🌎

## JinaBertConfig

[[autodoc]] JinaBertConfig
    - all

## JinaBert specific outputs

[[autodoc]] models.bert.modeling_jina_bert.JinaBertForPreTrainingOutput

[[autodoc]] models.bert.modeling_tf_jina_bert.TFJinaBertForPreTrainingOutput

[[autodoc]] models.bert.modeling_flax_jina_bert.FlaxJinaBertForPreTrainingOutput


<frameworkcontent>
<pt>

## JinaBertModel

[[autodoc]] JinaBertModel
    - forward

## JinaBertForPreTraining

[[autodoc]] JinaBertForPreTraining
    - forward

## JinaBertLMHeadModel

[[autodoc]] JinaBertLMHeadModel
    - forward

## JinaBertForMaskedLM

[[autodoc]] JinaBertForMaskedLM
    - forward

## JinaBertForNextSentencePrediction

[[autodoc]] JinaBertForNextSentencePrediction
    - forward

## JinaBertForSequenceClassification

[[autodoc]] JinaBertForSequenceClassification
    - forward

## JinaBertForMultipleChoice

[[autodoc]] JinaBertForMultipleChoice
    - forward

## JinaBertForTokenClassification

[[autodoc]] JinaBertForTokenClassification
    - forward

## JinaBertForQuestionAnswering

[[autodoc]] JinaBertForQuestionAnswering
    - forward

</pt>
</frameworkcontent>
