<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU inference

With some optimizations, it is possible to efficiently run large model inference on a CPU. One of these optimization techniques involves compiling the PyTorch code into an intermediate format for high-performance environments like C++. The other technique fuses multiple operations into one kernel to reduce the overhead of running each operation separately.

You'll learn how to use [BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) for faster inference, and how to convert your PyTorch code to [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html). If you're using an Intel CPU, you can also use [graph optimizations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features.html#graph-optimization) from [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/index.html) to boost inference speed even more. Finally, learn how to use 🤗 Optimum to accelerate inference with ONNX Runtime or OpenVINO (if you're using an Intel CPU).

## BetterTransformer

BetterTransformer accelerates inference with its fastpath (native PyTorch specialized implementation of Transformer functions) execution. The two optimizations in the fastpath execution are:

1. fusion, which combines multiple sequential operations into a single "kernel" to reduce the number of computation steps
2. skipping the inherent sparsity of padding tokens to avoid unnecessary computation with nested tensors

BetterTransformer also converts all attention operations to use the more memory-efficient [scaled dot product attention](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention).

<Tip>

BetterTransformer is not supported for all models. Check this [list](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models) to see if a model supports BetterTransformer.

</Tip>

Before you start, make sure you have 🤗 Optimum [installed](https://huggingface.co/docs/optimum/installation).

Enable BetterTransformer with the [`PreTrainedModel.to_bettertransformer`] method:

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", torch_dtype="auto")
```

## TorchScript

TorchScript is an intermediate PyTorch model representation that can be run in production environments where performance is important. You can train a model in PyTorch and then export it to TorchScript to free the model from Python performance constraints. PyTorch [traces](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) a model to return a [`ScriptFunction`] that is optimized with just-in-time compilation (JIT). Compared to the default eager mode, JIT mode in PyTorch typically yields better performance for inference using optimization techniques like operator fusion.

For a gentle introduction to TorchScript, see the [Introduction to PyTorch TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) tutorial.

With the [`Trainer`] class, you can enable JIT mode for CPU inference by setting the `--jit_mode_eval` flag:

```bash
python examples/pytorch/question-answering/run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--jit_mode_eval
```

<Tip warning={true}>

For PyTorch >= 1.14.0, JIT-mode could benefit any model for prediction and evaluation since the dict input is supported in `jit.trace`.

For PyTorch < 1.14.0, JIT-mode could benefit a model if its forward parameter order matches the tuple input order in `jit.trace`, such as a question-answering model. If the forward parameter order does not match the tuple input order in `jit.trace`, like a text classification model, `jit.trace` will fail and we are capturing this with the exception here to make it fallback. Logging is used to notify users.

</Tip>

## IPEX graph optimization

Intel® Extension for PyTorch (IPEX) provides further optimizations in JIT mode for Intel CPUs, and we recommend combining it with TorchScript for even faster performance. The IPEX [graph optimization](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html) fuses operations like Multi-head attention, Concat Linear, Linear + Add, Linear + Gelu, Add + LayerNorm, and more.

To take advantage of these graph optimizations, make sure you have IPEX [installed](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html):

```bash
pip install intel_extension_for_pytorch
```

Set the `--use_ipex` and `--jit_mode_eval` flags in the [`Trainer`] class to enable JIT mode with the graph optimizations:

```bash
python examples/pytorch/question-answering/run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--use_ipex \
--jit_mode_eval
```

## 🤗 Optimum

<Tip>

Learn more details about using ORT with 🤗 Optimum in the [Optimum Inference with ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models) guide. This section only provides a brief and simple example.

</Tip>

ONNX Runtime (ORT) is a model accelerator that runs inference on CPUs by default. ORT is supported by 🤗 Optimum which can be used in 🤗 Transformers, without making too many changes to your code. You only need to replace the 🤗 Transformers `AutoClass` with its equivalent [`~optimum.onnxruntime.ORTModel`] for the task you're solving, and load a checkpoint in the ONNX format.

For example, if you're running inference on a question answering task, load the [optimum/roberta-base-squad2](https://huggingface.co/optimum/roberta-base-squad2) checkpoint which contains a `model.onnx` file:

```py
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

model = ORTModelForQuestionAnswering.from_pretrained("optimum/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."
pred = onnx_qa(question, context)
```

If you have an Intel CPU, take a look at 🤗 [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) which supports a variety of compression techniques (quantization, pruning, knowledge distillation) and tools for converting models to the [OpenVINO](https://huggingface.co/docs/optimum/intel/inference) format for higher performance inference.
