<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BEiT

## Overview

The BEiT model was proposed in [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) by
Hangbo Bao, Li Dong and Furu Wei. Inspired by BERT, BEiT is the first paper that makes self-supervised pre-training of
Vision Transformers (ViTs) outperform supervised pre-training. Rather than pre-training the model to predict the class
of an image (as done in the [original ViT paper](https://arxiv.org/abs/2010.11929)), BEiT models are pre-trained to
predict visual tokens from the codebook of OpenAI's [DALL-E model](https://arxiv.org/abs/2102.12092) given masked
patches.

The abstract from the paper is the following:

*We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation
from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image
modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image
patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into
visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training
objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we
directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder.
Experimental results on image classification and semantic segmentation show that our model achieves competitive results
with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K,
significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains
86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%).*

This model was contributed by [nielsr](https://huggingface.co/nielsr). The JAX/FLAX version of this model was
contributed by [kamalkraj](https://huggingface.co/kamalkraj). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/beit).

## Usage tips

- BEiT models are regular Vision Transformers, but pre-trained in a self-supervised way rather than supervised. They
  outperform both the [original model (ViT)](vit) as well as [Data-efficient Image Transformers (DeiT)](deit) when fine-tuned on ImageNet-1K and CIFAR-100. You can check out demo notebooks regarding inference as well as
  fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) (you can just replace
  [`ViTFeatureExtractor`] by [`BeitImageProcessor`] and
  [`ViTForImageClassification`] by [`BeitForImageClassification`]).
- There's also a demo notebook available which showcases how to combine DALL-E's image tokenizer with BEiT for
  performing masked image modeling. You can find it [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BEiT).
- As the BEiT models expect each image to be of the same size (resolution), one can use
  [`BeitImageProcessor`] to resize (or rescale) and normalize images for the model.
- Both the patch resolution and image resolution used during pre-training or fine-tuning are reflected in the name of
  each checkpoint. For example, `microsoft/beit-base-patch16-224` refers to a base-sized architecture with patch
  resolution of 16x16 and fine-tuning resolution of 224x224. All checkpoints can be found on the [hub](https://huggingface.co/models?search=microsoft/beit).
- The available checkpoints are either (1) pre-trained on [ImageNet-22k](http://www.image-net.org/) (a collection of
  14 million images and 22k classes) only, (2) also fine-tuned on ImageNet-22k or (3) also fine-tuned on [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/) (also referred to as ILSVRC 2012, a collection of 1.3 million
  images and 1,000 classes).
- BEiT uses relative position embeddings, inspired by the T5 model. During pre-training, the authors shared the
  relative position bias among the several self-attention layers. During fine-tuning, each layer's relative position
  bias is initialized with the shared relative position bias obtained after pre-training. Note that, if one wants to
  pre-train a model from scratch, one needs to either set the `use_relative_position_bias` or the
  `use_relative_position_bias` attribute of [`BeitConfig`] to `True` in order to add
  position embeddings.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/beit_architecture.jpg"
alt="drawing" width="600"/>

<small> BEiT pre-training. Taken from the <a href="https://arxiv.org/abs/2106.08254">original paper.</a> </small>

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```
from transformers import BeitForImageClassification
model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (NVIDIA GeForce RTX 2060-8GB, PyTorch 2.5.1, OS Ubuntu 20.04) with `float16` and 
`microsoft/beit-base-patch16-224` model, we saw the following improvements during training and inference:

#### Training

| num_training_steps | batch_size | image_size   | is_cuda | Time per batch (eager - s) | Time per batch (sdpa - s) | Speedup (%) | Eager peak mem (MB) | SDPA peak mem (MB) | Mem saving (%) |
|--------------------|------------|--------------|---------|----------------------------|---------------------------|-------------|----------------------|--------------------|----------------|
| 50                 | 2          | (1048, 640)  | True    | 0.984                      | 0.746                     | 31.975      | 6738.915            | 4319.886          | 55.998         |

#### Inference

|   Image batch size |   Eager (s/iter) | Eager CI, %   |   Eager memory (MB) |   SDPA (s/iter) | SDPA CI, %   |   SDPA memory (MB) |   SDPA speedup | SDPA memory saved (%) |
|-------------------:|-----------------:|:--------------|--------------------:|----------------:|:-------------|-------------------:|---------------:|----------------------:|
|                  1 |            0.012 | ±0.3%         |         3.76657e+08 |           0.011 | ±0.5%        |        3.75739e+08 |          1.05  |                 0.244 |
|                  4 |            0.013 | ±0.1%         |         4.03147e+08 |           0.011 | ±0.2%        |        3.90554e+08 |          1.178 |                 3.225 |
|                 16 |            0.045 | ±0.1%         |         4.96697e+08 |           0.035 | ±0.1%        |        4.51232e+08 |          1.304 |                10.076 |
|                 32 |            0.088 | ±0.1%         |         6.24417e+08 |           0.066 | ±0.1%        |        5.33488e+08 |          1.325 |                17.044 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BEiT.

<PipelineTag pipeline="image-classification"/>

- [`BeitForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

**Semantic segmentation**
- [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## BEiT specific outputs

[[autodoc]] models.beit.modeling_beit.BeitModelOutputWithPooling

[[autodoc]] models.beit.modeling_flax_beit.FlaxBeitModelOutputWithPooling

## BeitConfig

[[autodoc]] BeitConfig

## BeitFeatureExtractor

[[autodoc]] BeitFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## BeitImageProcessor

[[autodoc]] BeitImageProcessor
    - preprocess
    - post_process_semantic_segmentation

<frameworkcontent>
<pt>

## BeitModel

[[autodoc]] BeitModel
    - forward

## BeitForMaskedImageModeling

[[autodoc]] BeitForMaskedImageModeling
    - forward

## BeitForImageClassification

[[autodoc]] BeitForImageClassification
    - forward

## BeitForSemanticSegmentation

[[autodoc]] BeitForSemanticSegmentation
    - forward

</pt>
<jax>

## FlaxBeitModel

[[autodoc]] FlaxBeitModel
    - __call__

## FlaxBeitForMaskedImageModeling

[[autodoc]] FlaxBeitForMaskedImageModeling
    - __call__

## FlaxBeitForImageClassification

[[autodoc]] FlaxBeitForImageClassification
    - __call__

</jax>
</frameworkcontent>