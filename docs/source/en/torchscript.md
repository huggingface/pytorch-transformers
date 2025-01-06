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

[TorchScript](https://pytorch.org/docs/stable/jit.html) serializes PyTorch models into programs that can be executed in non-Python processes. This is especially advantageous in production environments where Python may the most performant choice.

Transformers can export a model to TorchScript by:

1. creating dummy inputs to create a *trace* of the model to serialize to TorchScript
2. enabling the `torchscript` parameter in either [`~PretrainedConfig.torchscript`] for a randomly initialized model or [`~PreTrainedModel.from_pretrained`] for a pretrained model

## Dummy inputs

The dummy inputs are used in the forward pass, and as the input values are propagated through each layer, PyTorch tracks the different operations executed on each tensor. The recorded operations are used to create the model trace. Once it is recorded, it is serialized into a TorchScript program.

```py
from transformers import BertModel, BertTokenizer, BertConfig
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]
```

The trace is created based on the provided inputs dimensions and it can only handle inputs with the same shape as the provided input during tracing. An input with a different size raises the error message shown below.

```bash
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`.
```

Try to create a trace with a dummy input size at least as large as the largest expected input during inference. Padding can help fill missing values for larger inputs. It may be slower though since a larger input size requires more calculations. Be mindful of the total number of operations performed on each input and track the model performance when exporting models with variable sequence lengths.

## torchscript parameter

Weights between the `Embedding` and `Decoding` layers are tied in Transformers and TorchScript can't export models with tied weights. Instantiating a model with `torchscript=True`, separates the `Embedding` and `Decoding` layers and they aren't trained any further because it would throw the two layers out of sync which can lead to unexpected results.

Models *without* a language model head don't have tied weights and can be safely exported without the `torchscript` parameter.

<hfoptions id="torchscript">
<hfoption id="randomly initialized model">

```py
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

model = BertModel(config)
model.eval()
```

</hfoption>
<hfoption id="pretrained model">

```py
model = BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)
model.eval()
```

</hfoption>
</hfoptions>

## Export to TorchScript

Create the Torchscript program with [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html), and save with [torch.jit.save](https://pytorch.org/docs/stable/generated/torch.jit.save.html).

```py
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

Use [torch.jit.load](https://pytorch.org/docs/stable/generated/torch.jit.load.html) to load the traced model.

```py
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

To use the traced model for inference, use the `__call__` dunder method.

```py
traced_model(tokens_tensor, segments_tensors)
```

## Deploy to AWS

TorchScript programs serialized from Transformers can be deployed on [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) instances. The instance is powered by AWS Inferentia chips, a custom hardware accelerator designed for deep learning inference workloads. [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) supports tracing Transformers models for deployment on Inf1 instances.

> [!TIP]
> AWS Neuron requires a [Neuron SDK environment](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/inference-torch-neuron.html#inference-torch-neuron) which is preconfigured on [AWS DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html).

Instead of [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html), use [torch.neuron.trace](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/api-compilation-python-api.html) to trace a model and optimize it for Inf1 instances.

```py
import torch.neuron

torch.neuron.trace(model, [tokens_tensor, segments_tensors])
```

Refer to the [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html) documentation for more information.

### Model architectures

BERT-based models - like [DistilBERT](./model_doc/distilbert) or [RoBERTa](./model_doc/roberta) - run best on Inf1 instances for non-generative tasks such as extractive question answering, and sequence or token classification.

Text generation can be adapted to run on an Inf1 instance as shown in the [Transformers MarianMT](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html) tutorial.

Refer to the [Inference Samples/Tutorials (Inf1)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/models/inference-inf1-samples.html#model-samples-inference-inf1) guide for more information about which models can be converted out of the box to run on Inf1 instances.
