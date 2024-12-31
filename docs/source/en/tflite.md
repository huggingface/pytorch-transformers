<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LiteRT

[LiteRT](https://ai.google.dev/edge/litert) (previously known as TensorFlow Lite) is a high-performance runtime designed for on-device machine learning.

The [Optimum](https://huggingface.co/docs/optimum/index) library exports a model to LiteRT for [many architectures]((https://huggingface.co/docs/optimum/exporters/onnx/overview)).

The benefits of exporting to LiteRT include the following.

- Low-latency, privacy-focused, no internet connectivity required, and reduced model size and power consumption for on-device machine learning.
- Broad platform, model framework, and language support.
- Hardware acceleration for GPUs and Apple Silicon.

Export a Transformers model to LiteRT with the Optimum CLI.

Run the command below to install Optimum and the [exporters](https://huggingface.co/docs/optimum/exporters/overview) module for LiteRT.

```bash
pip install optimum[exporters-tf]
```

> [!TIP]
> Refer to the [Export a model to TFLite with optimum.exporters.tflite](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model) guide for all available arguments or with the command below.
> ```bash
> optimum-cli export tflite --help
> ```

Set the `--model` argument to export a from the Hub.

```bash
optimum-cli export tflite --model google-bert/bert-base-uncased --sequence_length 128 bert_tflite/
```

You should see logs indicating the progress and showing where the resulting `model.tflite` is saved.

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
 ```

For local models, make sure the model weights and tokenizer files are saved in the same directory, for example `local_path`. Pass the directory to the `--model` argument and use `--task` to indicate the [task](https://huggingface.co/docs/optimum/exporters/task_manager) a model can perform. If `--task` isn't provided, the model architecture without a task-specific head is used.

```bash
optimum-cli export tflite --model local_path --task question-answering google-bert/bert-base-uncased --sequence_length 128 bert_tflite/
```
