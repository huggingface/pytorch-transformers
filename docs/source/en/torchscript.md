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

# TorchScript

<Tip>

This is the very beginning of our experiments with TorchScript and we are still
exploring its capabilities with variable-input-size models. It is a focus of interest to
us and we will deepen our analysis in upcoming releases, with more code examples, a more
flexible implementation, and benchmarks comparing Python-based codes with compiled
TorchScript.

</Tip>

According to the [TorchScript documentation](https://pytorch.org/docs/stable/jit.html):

> TorchScript is a way to create serializable and optimizable models from PyTorch code.

There are two PyTorch modules, [JIT and
TRACE](https://pytorch.org/docs/stable/jit.html), that allow developers to export their
models to be reused in other programs like efficiency-oriented C++ programs.

We provide an interface that allows you to export 🤗 Transformers models to TorchScript
so they can be reused in a different environment than PyTorch-based Python programs.
Here, we explain how to export and use our models using TorchScript.

Exporting a model requires two things:

- model instantiation with the `torchscript` flag
- a forward pass with dummy inputs

These necessities imply several things developers should be careful about as detailed
below.

## TorchScript flag and tied weights

The `torchscript` flag is necessary because most of the 🤗 Transformers language models
have tied weights between their `Embedding` layer and their `Decoding` layer.
TorchScript does not allow you to export models that have tied weights, so it is
necessary to untie and clone the weights beforehand.

Models instantiated with the `torchscript` flag have their `Embedding` layer and
`Decoding` layer separated, which means that they should not be trained down the line.
Training would desynchronize the two layers, leading to unexpected results.

This is not the case for models that do not have a language model head, as those do not
have tied weights. These models can be safely exported without the `torchscript` flag.

## Dummy inputs and standard lengths

The dummy inputs are used for a models forward pass. While the inputs' values are
propagated through the layers, PyTorch keeps track of the different operations executed
on each tensor. These recorded operations are then used to create the *trace* of the
model.

The trace is created relative to the inputs' dimensions. It is therefore constrained by
the dimensions of the dummy input, and will not work for any other sequence length or
batch size. When trying with a different size, the following error is raised:

```
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`
```

We recommended you trace the model with a dummy input size at least as large as the
largest input that will be fed to the model during inference. Padding can help fill the
missing values. However, since the model is traced with a larger input size, the
dimensions of the matrix will also be large, resulting in more calculations.

Be careful of the total number of operations done on each input and follow the
performance closely when exporting varying sequence-length models.

## Using TorchScript in Python

This section demonstrates how to save and load models as well as how to use the trace
for inference.

### Saving a model

To export a `BertModel` with TorchScript, instantiate `BertModel` from the `BertConfig`
class and then save it to disk under the filename `traced_bert.pt`:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with *from_pretrained* you can also easily set the TorchScript flag
model = BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### Loading a model

Now you can load the previously saved `BertModel`, `traced_bert.pt`, from disk and use
it on the previously initialised `dummy_input`:

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### Using a traced model for inference

Use the traced model for inference by using its `__call__` dunder method:

```python
traced_model(tokens_tensor, segments_tensors)
```

## Deploy Hugging Face TorchScript models to AWS with the Neuron SDK

AWS introduced the [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/)
instance family for low cost, high performance machine learning inference in the cloud.
The Inf1 instances are powered by the AWS Inferentia chip, a custom-built hardware
accelerator, specializing in deep learning inferencing workloads. [AWS
Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) is the SDK for
Inferentia that supports tracing and optimizing transformers models for deployment on
Inf1. The Neuron SDK provides:


1. Easy-to-use API with one line of code change to trace and optimize a TorchScript
   model for inference in the cloud.
2. Out of the box performance optimizations for [improved
   cost-performance](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/>).
3. Support for Hugging Face transformers models built with either
   [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html)
   or
   [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html).

### Implications

Transformers models based on the [BERT (Bidirectional Encoder Representations from
Transformers)](https://huggingface.co/docs/transformers/main/model_doc/bert)
architecture, or its variants such as
[distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) and
[roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta) run best on
Inf1 for non-generative tasks such as extractive question answering, sequence
classification, and token classification. However, text generation tasks can still be
adapted to run on Inf1 according to this [AWS Neuron MarianMT
tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html).
More information about models that can be converted out of the box on Inferentia can be
found in the [Model Architecture
Fit](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia)
section of the Neuron documentation.

### Dependencies

Using AWS Neuron to convert models requires a [Neuron SDK
environment](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide)
which comes preconfigured on [AWS Deep Learning
AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html).

### Converting a model for AWS Neuron

Convert a model for AWS NEURON using the same code from [Using TorchScript in
Python](torchscript#using-torchscript-in-python) to trace a `BertModel`. Import the
`torch.neuron` framework extension to access the components of the Neuron SDK through a
Python API:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

You only need to modify the following line:

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [tokens_tensor, segments_tensors])
```

This enables the Neuron SDK to trace the model and optimize it for Inf1 instances.

To learn more about AWS Neuron SDK features, tools, example tutorials and latest
updates, please see the [AWS NeuronSDK
documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html).
