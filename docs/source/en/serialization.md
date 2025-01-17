<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ONNX

[ONNX](http://onnx.ai) is an open standard that defines a common set of operators and a file format to represent deep learning models in different frameworks, including PyTorch and TensorFlow. When a model is exported to ONNX, the operators construct a computational graph (or *intermediate representation*) which represents the flow of data through the model. Standardized operators and data types makes it easy to switch between frameworks.

The [Optimum](https://huggingface.co/docs/optimum/index) library exports a model to ONNX with configuration objects which are supported for [many architectures]((https://huggingface.co/docs/optimum/exporters/onnx/overview)) and can be easily extended. If a model isn't supported, feel free to make a [contribution](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute) to Optimum.

The benefits of exporting to ONNX include the following.

- [Graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) and [quantization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization) for improving inference.
- Use the [`~optimum.onnxruntime.ORTModel`] API to run a model with [ONNX Runtime](https://onnxruntime.ai/).
- Use [optimized inference pipelines](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines) for ONNX models.

Export a Transformers model to ONNX with the Optimum CLI or the `optimum.onnxruntime` module.

## Optimum CLI

Run the command below to install Optimum and the [exporters](https://huggingface.co/docs/optimum/exporters/overview) module.

```bash
pip install optimum[exporters]
```

> [!TIP]
> Refer to the [Export a model to ONNX with optimum.exporters.onnx](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli) guide for all available arguments or with the command below.
> ```bash
> optimum-cli export onnx --help
> ```

Set the `--model` argument to export a PyTorch or TensorFlow model from the Hub.

```bash
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

You should see logs indicating the progress and showing where the resulting `model.onnx` is saved.

```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
	-[✓] ONNX model output names match reference model (start_logits, end_logits)
	- Validating ONNX Model output "start_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
	- Validating ONNX Model output "end_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

For local models, make sure the model weights and tokenizer files are saved in the same directory, for example `local_path`. Pass the directory to the `--model` argument and use `--task` to indicate the [task](https://huggingface.co/docs/optimum/exporters/task_manager) a model can perform. If `--task` isn't provided, the model architecture without a task-specific head is used.

```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

The `model.onnx` file can be deployed with any [accelerator](https://onnx.ai/supported-tools.html#deployModel) that supports ONNX. The example below demonstrates loading and running a model with ONNX Runtime.

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

## optimum.onnxruntime

The `optimum.onnxruntime` module supports programmatically exporting a Transformers model. Instantiate a [`~optimum.onnxruntime.ORTModel`] for a task and set `export=True`. Use [`~OptimizedModel.save_pretrained`] to save the ONNX model.

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```
