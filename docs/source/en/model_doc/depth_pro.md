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

# DepthPro

## Overview

The DepthPro model was proposed in [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073) by Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun.

It leverages a multi-scale [Vision Transformer (ViT)](vit) optimized for dense predictions. It downsamples an image at several scales. At each scale, it is split into patches, which are processed by a ViT-based [Dinov2](dinov2) patch encoder, with weights shared across scales. Patches are merged into feature maps, upsampled, and fused via a [DPT](dpt)-like decoder.

The abstract from the paper is the following:

*We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions.*

<img src="https://huggingface.co/geetu040/DepthPro/resolve/main/assets/architecture.jpg"
alt="drawing" width="600"/>

<small> DepthPro architecture. Taken from the <a href="https://arxiv.org/abs/2410.02073" target="_blank">original paper</a>. </small>

This model was contributed by [geetu040](https://github.com/geetu040). The original code can be found [here](https://github.com/apple/ml-depth-pro).

<!-- TODO -->

## Usage tips

```python
from transformers import DepthProConfig, DepthProForDepthEstimation

config = DepthProConfig()
model = DepthProForDepthEstimation(config=config)
```

- Input image is scaled with different ratios, as specified in `scaled_images_ratios`, and each of the scaled image is patched to `patch_size` with an overlap ratio of `scaled_images_overlap_ratios`.
- These patches go through `DinoV2 (ViT)` based encoders and are reassembled via a `DPT` based decoder.
- `DepthProForDepthEstimation` can also predict the `FOV (Field of View)` if `use_fov_model` is set to `True` in the config.
- `DepthProImageProcessor` can be used for preprocessing the inputs and postprocessing the outputs. `DepthProImageProcessor.post_process_depth_estimation` interpolates the `predicted_depth` back to match the input image size.
- To generate `predicted_depth` of the same size as input image, make sure the config is created such that
```
image_size / 2**(n_fusion_blocks+1) == patch_size / patch_embeddings_size

where
n_fusion_blocks = len(intermediate_hook_ids) + len(scaled_images_ratios)
```


### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```py
from transformers import DepthProForDepthEstimation
model = DepthProForDepthEstimation.from_pretrained("geetu040/DepthPro", attn_implementation="sdpa", torch_dtype=torch.float16)
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `google/vit-base-patch16-224` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                         7 |                                         6 |                      1.17 |
|            2 |                                         8 |                                         6 |                      1.33 |
|            4 |                                         8 |                                         6 |                      1.33 |
|            8 |                                         8 |                                         6 |                      1.33 |

## Resources

- Research Paper: [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/pdf/2410.02073)

- Official Implementation: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)

<!-- A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DepthPro. -->

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DepthProConfig

[[autodoc]] DepthProConfig

## DepthProImageProcessor

[[autodoc]] DepthProImageProcessor
    - preprocess
    - post_process_depth_estimation

## DepthProImageProcessorFast

[[autodoc]] DepthProImageProcessorFast
    - preprocess
    - post_process_depth_estimation

## DepthProModel

[[autodoc]] DepthProModel
    - forward

## DepthProForDepthEstimation

[[autodoc]] DepthProForDepthEstimation
    - forward
