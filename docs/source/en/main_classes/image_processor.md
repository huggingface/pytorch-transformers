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

# Image Processor

An image processor is in charge of preparing input features for vision models and post processing their outputs. This includes transformations such as resizing, normalization, and conversion to PyTorch, TensorFlow, Flax and Numpy tensors. It may also include model specific post-processing such as converting logits to segmentation masks.

Fast image processors are available for a few models and more will be added in the future. They are based on the [torchvision](https://pytorch.org/vision/stable/index.html) library and provide a significant speed-up, especially when processing on GPU.
They have the same API as the base image processors and can be used as drop-in replacements.
To use a fast image processor, you need to install the `torchvision` library, and set the `use_fast` argument to `True` when instantiating the image processor:

```python
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
```
Note that `use_fast` will be set to `True` by default in a future release.

When using a fast image processor, you can also set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise.

```python
from torchvision.io import read_image
from transformers import DetrImageProcessorFast

images = read_image("image.jpg")
processor = DetrImageProcessorFast.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

Here are some speed comparisons between the base and fast image processors for the `DETR` and `RT-DETR` models, and how they impact overall inference time:

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_padded.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_batched_compiled.png" />
</div>

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_single.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_batched.png" />
</div>

These benchmarks were run on an [AWS EC2 g5.2xlarge instance](https://aws.amazon.com/ec2/instance-types/g5/), utilizing an NVIDIA A10G Tensor Core GPU.


## ImageProcessingMixin

[[autodoc]] image_processing_utils.ImageProcessingMixin
    - from_pretrained
    - save_pretrained

## BatchFeature

[[autodoc]] BatchFeature

## BaseImageProcessor

[[autodoc]] image_processing_utils.BaseImageProcessor


## BaseImageProcessorFast

[[autodoc]] image_processing_utils_fast.BaseImageProcessorFast
