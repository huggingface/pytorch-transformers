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

# AIMV2

## Overview

The AIMV2 model was proposed in [Multimodal Autoregressive Pre-training of Large Vision Encoders](https://arxiv.org/abs/2411.14402) by Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, and Alaaeldin El-Nouby.
AIMV2, a family of generalist vision encoders characterized by a straightforward pre-training process, scalability, and remarkable performance across a range of downstream tasks.

The abstract from the paper is the following:

*We introduce a novel method for pre-training of large-scale
vision encoders. Building on recent advancements in autoregressive pre-training of vision models, we extend this
framework to a multimodal setting, i.e., images and text. In
this paper, we present AIMV2, a family of generalist vision
encoders characterized by a straightforward pre-training
process, scalability, and remarkable performance across a
range of downstream tasks. This is achieved by pairing the
vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens. Our
encoders excel not only in multimodal evaluations but also
in vision benchmarks such as localization, grounding, and
classification. Notably, our AIMV2-3B encoder achieves
89.5% accuracy on ImageNet-1k with a frozen trunk. Furthermore, AIMV2 consistently outperforms state-of-the-art
contrastive models (e.g., CLIP, SigLIP) in multimodal image understanding across diverse settings.
*

Tips:

- The model is best suited for fine-tuning on downstream vision tasks such as image classification, object detection, and semantic segmentation.
- When using the model for inference, make sure to use an `AutoImageProcessor` (or manually process the images) to ensure the input images are preprocessed correctly (resized, normalized, etc.). The recommended image size for AIMv2 is typically 224x224, though some variants are trained on other resolutions (e.g., 336x336, 448x448). See the specific model checkpoint's documentation for details.
- AIMv2 models are trained using masked image modeling. If using the model for transfer learning, you may notice better performance by incorporating masked data during fine-tuning.

This model was contributed by [AlanPonnachan](https://huggingface.co/AlanPonnachan).
The original code can be found [here](https://github.com/apple/ml-aim).


## AIMv2Config

[[autodoc]] AIMv2Config

## AIMv2Model

[[autodoc]] AIMv2Model
    - forward


</pt>
<tf>
