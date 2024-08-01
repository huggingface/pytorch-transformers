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

# Load

Transformers provides many pretrained models that are ready to use with just a single line of code. It requires a model class and the [`~PreTrainedModel.from_pretrained`] method.

To load a model, call the [`~PreTrainedModel.from_pretrained`] method to download and load the model weights and configuration stored on the Hugging Face [Hub](https://hf.co/models) into the model class.

> [!TIP]
> The [`~PreTrainedModel.from_pretrained`] method loads weights stored in the [safetensors](https://hf.co/docs/safetensors/index) file format if they're available. Traditionally, PyTorch model weights are serialized with the [pickle](https://docs.python.org/3/library/pickle.html) utility which is known to be unsecure. Safetensor files are more secure and faster to load.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

This guide will briefly explain how models are loaded, the different ways you can load a model, and how to overcome memory issues for really big models.

## Models and configurations

All models have a `configuration.py` file with specific attributes like the number of hidden layers, vocabulary size, activation function, and more. You'll also find a `modeling.py` file that defines the layers and mathematical operations taking place inside each layer. The `modeling.py` file takes the model attributes in `configuration.py` and builds the model accordingly. At this point, you have a model with random weights that needs to be trained to output meaningful results.

<!-- insert diagram of model and configuration -->

> [!TIP]
> An *architecture* refers to the model's skeleton and a *checkpoint* refers to the model's weights for a given architecture. For example, [BERT](./model_doc/bert) is an architecture while [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) is a checkpoint. You'll see the term *model* used interchangeably for architecture and checkpoint.

To get a pretrained model, you need to load the weights into the model. This is done by calling the [`~PreTrainedModel.from_pretrained`] method which accepts weights from the Hugging Face Hub or a local directory.

There are two general types of models you can load:

1. A generic model class like [`LlamaModel`] or [`AutoModel`] that outputs hidden states.
2. A model class with a specific *head* attached to the generic model, like [`LlamaForCausalLM`] or [`AutoModelForCausalLM`], for performing specific tasks.

For each model type, there is a separate class for each machine learning framework (PyTorch, TensorFlow, Flax). Pick the corresponding prefix for the framework you're using.

<hfoptions id="backend">
<hfoption id="PyTorch">

```py
from transformers import AutoModelForCausalLM, MistralForCausalLM

# load with AutoClass or model-specific class
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

</hfoption>
<hfoption id="TensorFlow">

```py
from transformers import TFAutoModelForCausalLM, TFMistralForCausalLM

# load with AutoClass or model-specific class
model = TFAutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = TFMistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

</hfoption>
<hfoption id="Flax">

```py
from transformers import FlaxAutoModelForCausalLM, FlaxMistralForCausalLM

# load with AutoClass or model-specific class
model = FlaxAutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = FlaxMistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

</hfoption>
</hfoptions>

## AutoClass

<Youtube id="AhChOFRegn4"/>

The [AutoClass](./model_doc/auto) API is a convenient way to load an architecture without needing to know the exact model class name, because there are many architectures. It automatically selects the correct model class based on the configuration file. You only need to know the task and checkpoint you want to use.

The AutoClass makes it easy to switch between models or tasks, as long as the architecture is supported for a given task.

For example, the same model can be used for separate tasks.

```py
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

# use the same API for 3 different tasks
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForQuestionAnswering.from_pretrained("meta-llama/Llama-2-7b-hf")
```

In other cases, you want to quickly try out several models for a task.

```py
from transformers import AutoModelForCausalLM

# use the same API to load 3 different models
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
```

## Model-specific class

The [AutoClass](#autoclass) builds on top of model-specific classes. All model classes that support a specific task are mapped to their respective `AutoModelFor` task class.

But if you already know which model class you want to use, then you could use its model-specific class directly.

```py
from transformers import LlamaModel, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

## Big models

Large pretrained models require a lot of memory to load. The loading process involves:

1. creating a model with random weights
2. loading the pretrained weights
3. placing the pretrained weights on the model

You need enough memory to hold two copies of the model weights (random and pretrained) which may not be possible depending on your hardware. In distributed training environments, this is an even bigger challenge because each process loads a pretrained model.

Transformers reduces some of these memory-related challenges with fast initialization, sharded checkpoints, leveraging Accelerate's [Big Model Inference](https://hf.co/docs/accelerate/usage_guides/big_modeling) feature, and supporting lower bit data types.

### Fast initialization

A PyTorch model is instantiated with random weights, or "empty" tensors, that take up space in memory without filling it.

Transformers boosts loading speed and avoids random weight initialization with the [_fast_init](https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/modeling_utils.py#L2710) parameter if the pretrained weights are correctly initialized. This parameter is set to `True` by default.

### Sharded checkpoints

For big models with sharded checkpoints, each shard is loaded sequentially after the previous shard is loaded. This limits memory-usage to only the model size and the largest shard size.

Transformers' [`~PreTrainedModel.save_pretrained`] method automatically shards checkpoints larger than 10GB.

The `max_shard_size` parameter defaults to 5GB for each shard because it is easier to run on free-tier GPU instances without running out of memory.

For example, let's shard [BioMistral/BioMistral-7B](https://hf.co/BioMistral/BioMistral-7B).

```py
from transformers import AutoModel
import tempfile
import os

model = AutoModel.from_pretrained("biomistral/biomistral-7b")
with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    print(sorted(os.listdir(tmp_dir)))
```

Reload the sharded checkpoint with [`~PreTrainedModel.from_pretrained`].

```py
with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir)
    new_model = AutoModel.from_pretrained(tmp_dir)
```

Sharded checkpoints can also be directly loaded with the [`~transformers.modeling_utils.load_sharded_checkpoint`] method.

```py
from transformers.modeling_utils import load_sharded_checkpoint

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    load_sharded_checkpoint(model, tmp_dir)
```

#### Model metadata

Transformers' [`~PreTrainedModel.save_pretrained`] method creates an index file that maps parameter names to the files they're stored in. The index file has two keys, `metadata` and `weight_map`.

```py
import json

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

print(index.keys())
```

The `metadata` key provides the total model size.

```py
index["metadata"]
{'total_size': 28966928384}
```

The `weight_map` key maps each parameter to the shard it's stored in.

```py
index["weight_map"]
{'lm_head.weight': 'model-00006-of-00006.safetensors',
 'model.embed_tokens.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.input_layernorm.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.mlp.down_proj.weight': 'model-00001-of-00006.safetensors',
 ...
}
```

### Big Model Inference

> [!TIP]
> Make sure you have Accelerate v0.9.0 and PyTorch v1.9.0 or later installed to use this feature!

<Youtube id="MWCSGj9jEAo"/>

The [`~PreTrainedModel.from_pretrained`] method is supercharged with Accelerate's [Big Model Inference](https://hf.co/docs/accelerate/usage_guides/big_modeling).

Big Model Inference creates a *model skeleton* on PyTorch's [meta](https://pytorch.org/docs/main/meta.html) device. The meta device doesn't store any real data, only the metadata.

Randomly initialized weights are only created when the pretrained weights are loaded to avoid maintaining two copies of the model in memory at the same time. The maximum memory-usage is only the size of the model.

> [!TIP]
> Learn more about device placement in [Designing a device map](https://hf.co/docs/accelerate/v0.33.0/en/concept_guides/big_model_inference#designing-a-device-map) section.

Big Model Inference's second feature relates to how weights are loaded and dispatched in the model skeleton. Model weights are dispatched across all available devices, starting with the fastest device (usually the GPU) and then offloading any remaining weights to slower devices (CPU and hard drive).

Both features combine reduced memory-usage and faster loading times for big pretrained models.

Set the [device_map](https://github.com/huggingface/transformers/blob/026a173a64372e9602a16523b8fae9de4b0ff428/src/transformers/modeling_utils.py#L3061) parameter to `"auto"` to enable Big Model Inference. This also sets the [low_cpu_mem_usage](https://github.com/huggingface/transformers/blob/026a173a64372e9602a16523b8fae9de4b0ff428/src/transformers/modeling_utils.py#L3028) parameter to `True`.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
```

To manually assign layers to devices, create a `device_map`. It should map all model parameters to a device, but you don't have to detail where all the submodules of a layer go if the entire layer is on the same device. Access the `hf_device_map` attribute to see how the model is distributed across devices.

```py
device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
model.hf_device_map
```

### Model data type

PyTorch model weights are initialized as torch.float32. To load a model in a different data type, like torch.float16, it requires additional memory to load the model again in the desired data type.

Explicitly set the [torch_dtype]() parameter to directly initialize the model in the desired data type instead of essentially loading a model twice (torch.float32, torch.float16). You could also set `torch_dtype="auto"` to automatically load the weights with the most optimal memory pattern (the data type is derived from the model weights).

<hfoptions id="dtype">
<hfoption id="specific dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.float16)
```

</hfoption>
<hfoption id="auto dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype="auto")
```

</hfoption>
</hfoptions>

The `torch_dtype` parameter can also be configured in [`AutoConfig`] for models instantiated from scratch.

```py
import torch
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("google/gemma-2b", torch_dtype=torch.float16)
model = AutoModel.from_config(my_config)
```

## Custom models

Custom models use Transformers' configuration and modeling classes, supports the [AutoClass](#autoclass) API, and are loaded with [`~PreTrainedModel.from_pretrained`]. What makes custom models different is the modeling code is not from Transformers.

The Hub includes [malware scanning](https://hf.co/docs/hub/security-malware#malware-scanning) for every repository, but extra care should still be taken when loading a custom model to avoid inadvertently executing malicious code.

Set the `trust_remote_code` parameter to `True` in [`~PreTrainedModel.from_pretrained`] to load a custom model.

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

As an extra layer of security, load a custom model from a specific revision to make sure the model code hasn't changed. The commit hash can be copied from the model's [commit history](https://hf.co/sgugger/custom-resnet50d/commits/main).

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

Learn more about how to create a custom model in [Customize](./custom_models).