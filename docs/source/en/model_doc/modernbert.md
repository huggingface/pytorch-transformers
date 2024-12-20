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

# ModernBert

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=modernbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-modernbert-blueviolet">
</a>
<a href="https://arxiv.org/abs/2412.13663">
<img alt="Paper page" src="https://img.shields.io/badge/Paper%20page-2412.13663-green">
</a>
</div>

## Overview

The ModernBert model was proposed in [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663) by Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Galalgher, Raja Bisas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Grifin Adams, Jeremy Howard and Iacopo Poli.

It is a refresh of the traditional encoder architecture, as used in previous models such as [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) and [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta). 

It builds on BERT and implements many modern architectural improvements which have been developed since its original release, such as:
- [Rotary Positional Embeddings](https://huggingface.co/blog/designing-positional-encoding) to support sequences of up to 8192 tokens.
- [Unpadding](https://arxiv.org/abs/2208.08124) to ensure no compute is wasted on padding tokens, speeding up processing time for batches with mixed-length sequences.
- [GeGLU](https://arxiv.org/abs/2002.05202) Replacing the original MLP layers with GeGLU layers, shown to improve performance.
- [Alternating Attention](https://arxiv.org/abs/2004.05150v2) where most attention layers employ a sliding window of 128 tokens, with Global Attention only used every 3 layers.
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) to speed up processing.
- A model designed following recent [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489), ensuring maximum efficiency across inference GPUs.
- Modern training data scales (2 trillion tokens) and mixtures (including code ande math data)

The abstract from the paper is the following:

*Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs.*

The original code can be found [here](https://github.com/answerdotai/modernbert).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ModernBert.

<PipelineTag pipeline="text-classification"/>

- A notebook on how to [finetune for General Language Understanding Evaluation (GLUE) with Transformers](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb), also available as a Google Colab [notebook](https://colab.research.google.com/github/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb). 🌎

<PipelineTag pipeline="sentence-similarity"/>

- A script on how to [finetune for text similarity or information retrieval with Sentence Transformers](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_st.py). 🌎
- A script on how to [finetune for information retrieval with PyLate](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_pylate.py). 🌎

<PipelineTag pipeline="fill-mask"/>

- [Masked language modeling task guide](../tasks/masked_language_modeling)


## ModernBertConfig

[[autodoc]] ModernBertConfig

<frameworkcontent>
<pt>

## ModernBertModel

[[autodoc]] ModernBertModel
    - forward

## ModernBertForMaskedLM

[[autodoc]] ModernBertForMaskedLM
    - forward

## ModernBertForSequenceClassification

[[autodoc]] ModernBertForSequenceClassification
    - forward

## ModernBertForTokenClassification

[[autodoc]] ModernBertForTokenClassification
    - forward

</pt>
</frameworkcontent>
