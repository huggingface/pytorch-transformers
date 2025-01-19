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

# DAB-DETR

## Overview

The DAB-DETR model was proposed in [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329) by Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, Lei Zhang.
DAB-DETR is an enhanced variant of Conditional DETR. It utilizes dynamically updated anchor boxes to provide both a reference query point (x, y) and a reference anchor size (w, h), improving cross-attention computation. This new approach achieves 45.7% AP when trained for 50 epochs with a single ResNet-50 model as the backbone.

<img src="https://github.com/conditionedstimulus/hf_media/blob/main/dab_detr_convergence_plot.png"
alt="drawing" width="600"/>

The abstract from the paper is the following:

*We present in this paper a novel query formulation using dynamic anchor boxes
for DETR (DEtection TRansformer) and offer a deeper understanding of the role
of queries in DETR. This new formulation directly uses box coordinates as queries
in Transformer decoders and dynamically updates them layer-by-layer. Using box
coordinates not only helps using explicit positional priors to improve the query-to-feature similarity and eliminate the slow training convergence issue in DETR,
but also allows us to modulate the positional attention map using the box width
and height information. Such a design makes it clear that queries in DETR can be
implemented as performing soft ROI pooling layer-by-layer in a cascade manner.
As a result, it leads to the best performance on MS-COCO benchmark among
the DETR-like detection models under the same setting, e.g., AP 45.7% using
ResNet50-DC5 as backbone trained in 50 epochs. We also conducted extensive
experiments to confirm our analysis and verify the effectiveness of our methods.*

This model was contributed by [davidhajdu](https://huggingface.co/davidhajdu).
The original code can be found [here](https://github.com/IDEA-Research/DAB-DETR).

There are three ways to instantiate a DAB-DETR model (depending on what you prefer):

Option 1: Instantiate DAB-DETR with pre-trained weights for entire model
```py
>>> from transformers import DabDetrForObjectDetection

>>> model = DabDetrForObjectDetection.from_pretrained("IDEA-Research/dab_detr_resnet50")
```

Option 2: Instantiate DAB-DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone
```py
>>> from transformers import DabDetrConfig, DabDetrForObjectDetection

>>> config = DabDetrConfig()
>>> model = DabDetrForObjectDetection(config)
```
Option 3: Instantiate DAB-DETR with randomly initialized weights for backbone + Transformer
```py
>>> config = DabDetrConfig(use_pretrained_backbone=False)
>>> model = DabDetrForObjectDetection(config)
```


## DabDetrConfig

[[autodoc]] DabDetrConfig

## DabDetrModel

[[autodoc]] DabDetrModel
    - forward

## DabDetrForObjectDetection

[[autodoc]] DabDetrForObjectDetection
    - forward
