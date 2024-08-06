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

# Customize

Transformers models are easily customizable. Models are fully contained in the [model](https://github.com/huggingface/transformers/tree/main/src/transformers/models) subfolder of the Transformers repository. Each folder contains a `modeling.py` and a `configuration.py` file. Copy these files to start customizing a model.

> [!TIP]
> It may be easier to start from scratch if you're creating an entirely new model. For models that are very similar to an existing one in Transformers, it is faster to reuse or subclass the same configuration and model class.

This guide will show you how to customize a ResNet model, enable [AutoClass](./models#autoclass) API support, and share it on the Hub.

## Configuration

A configuration, given by the base [`PretrainedConfig`] class, contains all the necessary information to build a model. This is where you'll configure the parameters of the custom ResNet model. Different configurations gives different ResNet model types.

The three main rules for customizing a configuration are:

1. A custom configuration must inherit from [`PretrainedConfig`]. Inheritance ensures a custom model has all the functionality of a Transformers model such as [`PretrainedConfig.from_pretrained`], [`PretrainedConfig.save_pretrained`], and [`PretrainedConfig.push_to_hub`].
2. The [`PretrainedConfig`] `__init__` must accept any `kwargs` and `kwargs` must be passed to the superclass `__init__`. [`PretrainedConfig`] has more more fields than the ones you're setting in your custom configuration. When you load a configuration with [`PretrainedConfig.from_pretrained`], those fields need to be accepted by your configuration and passed to the superclass.

> [!TIP]
> It is useful to check the validity of some of the parameters. In the example below, a check is implemented to ensure `block_type` and `stem_type` are one of the predefined values.
>
> Add `model_type` to the configuration class to enable [AutoClass](./models#autoclass) support.

```py
from transformers import PretrainedConfig
from typing import List

class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

Save the configuration to a JSON file with the [`PretrainedConfig.save_pretrained`] method. This file is stored in your custom model folder, `custom-resnet`.

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

## Model

With the custom ResNet configuration, you can now create and customize the model. The model inherits from the base [`PreTrainedModel`] class. Like [`PretrainedConfig`], inheriting from [`PreTrainedModel`] and initializing the superclass with the configuration extends Transformers functionalities such as saving and loading to the custom model.

Transformers' models follow the convention of accepting a `config` object in the `__init__` method. This passes the entire `config` to the models sublayers, instead of breaking the `config` object into multiple arguments that are passed individually to the sublayers. Writing models this way produces simpler code with a clear *source of truth* for any hyperparameters. It is also easier to reuse code from other Transformers' models.

You'll create two ResNet models, a ResNet model that outputs the hidden states and a ResNet model with an image classification head.

<hfoptions id="resnet">
<hfoption id="ResnetModel">

Define a mapping between the block types and block classes. Everything else is created by passing the configuration class to the Resnet model class.

> [!TIP]
> Add `config_class` to the model class to enable [AutoClass](#autoclass-support) support.

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig

BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

</hfoption>
<hfoption id="ResnetModelForImageClassification">

The `forward` method needs to be rewrittten to calculate the loss for each logit if labels are available. Otherwise, the Resnet model class is the same.

> [!TIP]
> Add `config_class` to the model class to enable [AutoClass](#autoclass-support) support.

```py
import torch

class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

</hfoption>
</hfoptions>

A model can return any output format. Returning a dictionary (like ResnetModelForImageClassification) with losses when labels are available, makes the custom model compatible with the [`Trainer`]. For other output formats, you'll need your own training loop or a different library for training.

Instantiate the custom model class with the configuration.

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

At this point, you can load pretrained weights into the model or train it from scratch. You'll load pretrained weights in this guide.

Load the pretrained weights from the [timm](https://hf.co/docs/timm/index) library, and then transfer those weights to the custom model with the [load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) method.

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

## AutoClass support

The [AutoClass](./models#autoclass) API is a shortcut for automatically loading the correct architecture for a given model. It may be convenient for your users to add this API to your custom model.

Make sure you have the `model_type` attribute (must be different from existing model types) in the configuration class and `config_class` attribute in the model class. With the [`~AutoConfig.register`] method, add the custom configuration and model to the [AutoClass](./models#autoclass) API.

> [!TIP]
> The first argument to [`AutoConfig.register`] must match the `model_type` attribute in the custom configuration class, and the first argument to [`AutoModel.register`] must match the `config_class` of the custom model class.

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

Your custom model code is now compatible with the [AutoClass](./models#autoclass) API. Users can load the model with the `AutoModel` or [`AutoModelForImageClassification`] classes.

## Upload model

Upload a custom model to the [Hub](https://hf.co/models) to allow other users to easily load and use it.

Ensure the model directory is structured correctly as shown below. The directory should contain:

- `modeling.py`: Contains the code for ResnetModel and ResnetModelForImageClassification. This file can rely on relative imports to other files as long as they're in the same directory.

> [!WARNING]
> Replace all relative imports at the top of the `modeling.py` file to import from Transformers instead if you're copying a model file from Transformers.

- `configuration.py`: Contains the code for ResnetConfig.
- `__init__.py`: Can be empty. This file allows Python `resnet_model` to be used as a module.

```bash
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

To share the model, import the ResNet model and configuration.

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

Copy the code from the model and configuration files. To make sure the AutoClass objects are saved when calling [`~PreTrainedModel.save_pretrained`], call the [`~PretrainedConfig.register_for_auto_class`] method. This modifies the configuration JSON file to include the AutoClass objects and mapping.

For a model, pick the appropriate `AutoModelFor` class based on the task.

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

To map more than one task to the model, edit `auto_map` in the configuration JSON file directly.

```json
"auto_map": {
    "AutoConfig": "<your-repo-name>--<config-name>",
    "AutoModel": "<your-repo-name>--<config-name>",
    "AutoModelFor<Task>": "<your-repo-name>--<config-name>",    
},
```

Create the configuration and model and load pretrained weights into it.

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

The model is ready to be pushed to the Hub now. Login to your Hugging Face account from the command line or notebook.

<hfoptions id="push">
<hfoption id="huggingface-CLI">

```bash
huggingface-cli login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

Call [`~PreTrainedModel.push_to_hub`] on the model to upload the model to the Hub.

```py
resnet50d.push_to_hub("custom-resnet50d")
```

The pretrained weights, configuration in JSON format, `modeling.py` and `configuration.py` files should all be uploaded to the Hub now under a namespace and specified directory [here](https://hf.co/sgugger/custom-resnet50d).

Because a custom model doesn't use the same modeling code as Transformers' model, you need to add `trust_remode_code=True` in the [`~PreTrainedModel.from_pretrained`] method. Refer to the load [custom models](./models#custom-models) section for more information.
