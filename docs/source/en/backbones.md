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

# Backbones

Higher-level computer visions tasks, such as object detection or image segmentation, use several models together to generate a prediction. A separate model is used for the *backbone*, neck, and head. The backbone extracts useful features from an input image into a feature map, the neck combines and processes the feature maps, and the head uses them to make a prediction.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Backbone.png"/>
</div>

Load a backbone with [`~PretrainedConfig.from_pretrained`] and use the `out_indices` parameter to determine which layer, given by the index, to extract a feature map from.

```py
from transformers import AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

This guide describes the backbone class, backbones from the [timm](https://hf.co/docs/timm/index) library, and how to extract features with them.

## Backbone classes

There are two backbone classes.

- [`~transformers.utils.BackboneMixin`] allows you to load a backbone and includes functions for extracting the feature maps and indices.
- [`~transformers.utils.BackboneConfigMixin`] allows you to set the feature map and indices of a backbone configuration.

Refer to the [Backbone](./main_classes/backbones) API documentation to check which models support a backbone.

There are two ways to load a Transformers backbone, [`AutoBackbone`] and a model-specific backbone class.

<hfoptions id="backbone-classes">
<hfoption id="AutoBackbone">

The [AutoClass](./model_doc/auto) API automatically loads a pretrained vision model with [`~PretrainedConfig.from_pretrained`] as a backbone if it's supported.

Set the `out_indices` parameter to the layer you'd like to get the feature map from. If you know the name of the layer, you could also use `out_features`. These parameters can be used interchangeably, but if you use both, make sure they refer to the same layer.

When `out_indices` or `out_features` isn't used, the backbone returns the feature map from the last layer. The example code below uses `out_indices=(1,)` to get the feature map from the first layer.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stage%201.png"/>
</div>

```py
from transformers import AutoImageProcessor, AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

</hfoption>
<hfoption id="model-specific backbone">

When you know a model supports a backbone, you can load the backbone and neck directly into the models configuration. Pass the configuration to the model to initialize it for a task.

The example below loads a [ResNet](./model_doc/resnet) backbone and neck for use in a [MaskFormer](./model_doc/maskformer) instance segmentation head.

Set `backbone` to a pretrained model and  `use_pretrained_backbone=True` to use pretrained weights instead of randomly initialized weights.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="microsoft/resnet-50", use_pretrained_backbone=True)
model = MaskFormerForInstanceSegmentation(config)
```

Another option is to separately load the backbone configuration and then pass it to `backbone_config` in the model configuration.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig

# instantiate backbone configuration
backbone_config = ResNetConfig()
# load backbone in model
config = MaskFormerConfig(backbone_config=backbone_config)
# attach backbone to model head
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
</hfoptions>

## timm backbones

[timm](https://hf.co/docs/timm/index) is a collection of vision models for training and inference. Transformers supports timm models as backbones with the [`TimmBackbone`] and [`TimmBackboneConfig`] classes.

Set `use_timm_backbone=True` to load pretrained timm weights, and `use_pretrained_backbone` to use pretrained or randomly initialized weights.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="resnet50", use_timm_backbone=True, use_pretrained_backbone=True)
model = MaskFormerForInstanceSegmentation(config)
```

You could also explicitly call the [`TimmBackboneConfig`] class to load and create a pretrained timm backbone.

```py
from transformers import TimmBackboneConfig

backbone_config = TimmBackboneConfig("resnet50", use_pretrained_backbone=True)
```

Pass the backbone configuration to the model configuration and instantiate the model head, [`MaskFormerForInstanceSegmentation`], with the backbone.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

## Feature extraction

The backbone is used to extract image features. Pass an image through the backbone to get the feature maps.

Load and preprocess an image and pass it to the backbone. The example below extracts the feature maps from the first layer.

```py
from transformers import AutoImageProcessor, AutoBackbone
import torch
from PIL import Image
import requests

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
```

The features are stored and accessed from the outputs `feature_maps` attribute.

```py
feature_maps = outputs.feature_maps
list(feature_maps[0].shape)
[1, 96, 56, 56]
```
