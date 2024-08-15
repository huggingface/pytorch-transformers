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

For some higher-level computer visions tasks such as object detection or image segmentation, it is common to use several models together to generate a prediction. These networks combine a *backbone*, neck, and head. The backbone extracts useful features from an input image into a feature map, the neck combines and processes the feature maps, and the head uses them to make a prediction.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Backbone.png"/>
</div>

Load a backbone with the [`~AutoBackbone.from_pretrained`] method.

```py
from transformers import AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

## Base backbone classes

There are two backbone classes for Transformers' models.

- [`BackboneMixin`] allows you to load a backbone and includes functions for extracting the feature maps and indices.
- [`BackboneConfigMixin`] allows you to set the feature map and indices of a backbone configuration.

Refer to the [Backbone](./main_classes/backbones) API documentation to check which models support a backbone.

## AutoBackbone

The [AutoClass](./model_doc/auto) API automatically loads a pretrained vision model with [`~AutoBackbone.from_pretrained`] as a backbone if it's supported.

Set the `out_indices` parameter to the layer you'd like to get the feature map from. If you known the name of the layer, you could also use `out_features`. These parameters can be used interchangeably, but if you use both, make sure they're referring to the same layer.

When you don't use `out_indices` or `out_features`, the backbone returns the feature map from the last layer. Specify `out_indices=(1,)` to get the feature map from the first layer.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stage%201.png"/>
</div>

```py
from transformers import AutoImageProcessor, AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

## Model-specific backbones

When you know a model supports a backbone, you can load the backbone and neck directly into the model's configuration. Then pass the configuration to the model to initialize it for a task.

For example, load a [ResNet](./model_doc/resnet) backbone and neck for use in a [MaskFormer](./model_doc/maskformer) instance segmentation head.

Set the `backbone` parameter to the pretrained model to load the model configuration class. Toggle the `use_pretrained_backbone` parameter to determine whether you want to use pretrained or randomly initialized weights.

<hfoptions id="backbone">
<hfoption id="pretrained weights">

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="microsoft/resnet-50", use_pretrained_backbone=True)
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
<hfoption id="random weights">

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="microsoft/resnet-50", use_pretrained_backbone=False)
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
</hfoptions>

Another option is to separately load the backbone configuration and then pass it to the `backbone_config` paramater in the model configuration.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig

# instantiate backbone configuration
backbone_config = ResNetConfig()
# load backbone in model
config = MaskFormerConfig(backbone_config=backbone_config)
# attach backbone to model head
model = MaskFormerForInstanceSegmentation(config)
```

## timm backbones

[timm](https://hf.co/docs/timm/index) is a collection of vision models for training and inference. Transformers supports timm models as backbones with the [`TimmBackbone`] and [`TimmBackboneConfig`] classes.

Set `use_timm_backnoe=True` to load pretrained timm weights. The `use_pretrained_backbone` parameter can be toggled to use pretrained or randomly initialized weights.

<hfoptions id="timm">
<hfoption id="pretrained weights">

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="resnet50", use_pretrained_backbone=True, use_timm_backbone=True)
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
<hfoption id="random weights">

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="resnet50", use_pretrained_backbone=False, use_timm_backbone=True)
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
</hfoptions>

You could also explicitly call the [`TimmBackboneConfig`] class to load and create a pretrained timm backbone.

```py
from transformers import TimmBackboneConfig

backbone_config = TimmBackboneConfig("resnet50", use_pretrained_backbone=True)
```

Pass the backbone configuration to the model configuration and then instantiate the model head, [`MaskFomerForInstanceSegmentation`], with the backbone.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

## Feature extraction

The backbone is used for image feature extraction. Pass an image through the backbone to get the feature maps.

Load and preprocess an image, and then pass it to the backbone.

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
