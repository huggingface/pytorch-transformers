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

# Deformable DETR

## Overview

The Deformable DETR model was proposed in [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.
Deformable DETR mitigates the slow convergence issues and limited feature spatial resolution of the original [DETR](detr) by leveraging a new deformable attention module which only attends to a small set of key sampling points around a reference.

The abstract from the paper is the following:

*DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deformable_detr_architecture.png"
alt="drawing" width="600"/>

<small> Deformable DETR architecture. Taken from the <a href="https://arxiv.org/abs/2010.04159">original paper</a>.</small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/fundamentalvision/Deformable-DETR).

## Usage tips

- Training Deformable DETR is equivalent to training the original [DETR](detr) model. See the [resources](#resources) section below for demo notebooks.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Deformable DETR.

<PipelineTag pipeline="object-detection"/>

- Demo notebooks regarding inference + fine-tuning on a custom dataset for [`DeformableDetrForObjectDetection`] can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR).
- Scripts for finetuning [`DeformableDetrForObjectDetection`] with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DeformableDetrImageProcessor

[[autodoc]] DeformableDetrImageProcessor
    - preprocess
    - post_process_object_detection

## DeformableDetrImageProcessorFast

[[autodoc]] DeformableDetrImageProcessorFast
    - preprocess
    - post_process_object_detection

## DeformableDetrFeatureExtractor

[[autodoc]] DeformableDetrFeatureExtractor
    - __call__
    - post_process_object_detection

## DeformableDetrConfig

[[autodoc]] DeformableDetrConfig

## DeformableDetrModel

[[autodoc]] DeformableDetrModel
    - forward

## DeformableDetrForObjectDetection

[[autodoc]] DeformableDetrForObjectDetection
    - forward
