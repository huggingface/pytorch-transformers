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

# Prompt Depth Anything

## Overview

The Prompt Depth Anything model was introduced in [Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](https://arxiv.org/abs/2412.14015) by Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, Bingyi Kang. 


The abstract from the paper is as follows:

*Prompts play a critical role in unleashing the power of language and vision foundation models for specific tasks. For the first time, we introduce prompting into depth foundation models, creating a new paradigm for metric depth estimation termed Prompt Depth Anything. Specifically, we use a low-cost LiDAR as the prompt to guide the Depth Anything model for accurate metric depth output, achieving up to 4K resolution. Our approach centers on a concise prompt fusion design that integrates the LiDAR at multiple scales within the depth decoder. To address training challenges posed by limited datasets containing both LiDAR depth and precise GT depth, we propose a scalable data pipeline that includes synthetic data LiDAR simulation and real data pseudo GT depth generation. Our approach sets new state-of-the-arts on the ARKitScenes and ScanNet++ datasets and benefits downstream applications, including 3D reconstruction and generalized robotic grasping.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/prompt_depth_anything_architecture.jpg"
alt="drawing" width="600"/>

<small> Prompt Depth Anything overview. Taken from the <a href="https://arxiv.org/pdf/2412.14015">original paper</a>.</small>

## Usage example

The Transformers library allows you to use the model with just a few lines of code:

```python
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
>>> model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")

>>> prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
>>> prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)
>>> # the prompt depth can be None, and the model will output a monocular relative depth.

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_sizes=[(image.height, image.width)],
... )

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 1000 
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint16")) # mm
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Prompt Depth Anything.

- [Prompt Depth Anything Demo](https://huggingface.co/spaces/depth-anything/PromptDA)
- [Prompt Depth Anything Interactive Results](https://promptda.github.io/interactive.html)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## PromptDepthAnythingConfig

[[autodoc]] PromptDepthAnythingConfig

## PromptDepthAnythingForDepthEstimation

[[autodoc]] PromptDepthAnythingForDepthEstimation
    - forward

## PromptDepthAnythingImageProcessor

[[autodoc]] PromptDepthAnythingImageProcessor
    - preprocess