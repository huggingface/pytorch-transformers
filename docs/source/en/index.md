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

# Transformers

Transformers is a library of pretrained natural language processing, computer vision, audio, and multimodal models for inference and training. Use Transformers to train models on your data, build inference applications, and generate text with large language models.

Explore the [Hugging Face Hub](https://huggingface.com) today to find a model and use Transformers to help you get started right away.

## Features

Transformers provides everything you need for inference or training with state-of-the-art pretrained models. Some of the main features include:

- [Pipeline](./pipeline_tutorial): Simple and optimized inference class for many machine learning tasks like text generation, image segmentation, automatic speech recognition, document question answering, and more.
- [Trainer](./trainer): A comprehensive trainer that supports features such as mixed precision, torch.compile, and FlashAttention for training and distributed training for PyTorch models.
- [generate](./llm_tutorial): Fast text generation with large language models (LLMs) and vision language models (VLMs), including support for streaming and multiple decoding strategies.

## Design

> [!TIP]
> Read our [Philosophy](./philosophy) to learn more about Transformers' design principles.

Transformers is designed for developers and machine learning engineers and researchers. Its main design principles are:

1. Fast and easy to use: Every model is implemented from only three main classes (configuration, model, and preprocessor) and can be quickly used for inference or training with [`Pipeline`] or [`Trainer`].
2. Pretrained models: Reduce your carbon footprint, compute cost and time by using a pretrained model instead of training an entirely new one. Each pretrained model is reproduced as closely as possible to the original model and offers state-of-the-art performance.

<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

Join us on the Hugging Face [Hub](https://huggingface.co/), [Discord](https://discord.com/invite/JfAtkvEtRb), or [forum](https://discuss.huggingface.co/) to collaborate and build models, datasets, and applications together.
