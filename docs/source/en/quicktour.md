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

# Quickstart

[[open-in-colab]]

Get up and running with Transformers! Whether you're a developer or a machine learning engineer, this quickstart will show you Transformers' key features.

Transformers is a library of pretrained models, providing three classes to instantiate any model and two APIs for inference or training. By limiting the number of user-facing abstractions, Transformers is easier to learn and faster to use.

Transformers supports popular machine learning frameworks like [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [Flax](https://flax.readthedocs.io/en/latest/). Switching between frameworks is easy, granting the flexibility to use the best tool for the job (training, evaluation, or production).

In this quickstart, you'll learn how to:

- load a pretrained model for inference or training
- run inference with the [`Pipeline`] API
- train a model with the [`Trainer`] API

## Setup

To start, we recommend creating a Hugging Face [account](https://hf.co/join). This allows you to host and access version controlled models, datasets, and apps on the [Hugging Face Hub](https://hf.co/docs/hub/index), a collaborative platform for discovery and building.

Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) and login to your account.

```py
from huggingface_hub import notebook_login

notebook_login()
```

Make sure your preferred machine learning framework is installed.

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
!pip install torch
```

</hfoption>
<hfoption id="TensorFlow">

```bash
!pip install tensorflow
```

</hfoption>
</hfoptions>

Install an up-to-date version of Transformers and some additional libraries from the Hugging Face ecosystem for accessing datasets and vision models, evaluating training, and optimizing training for large models.

```bash
!pip install -U transformers datasets evaluate accelerate timm
```

## Base classes

Each pretrained model inherits from three base classes.

| **Class** | **Description** |
|---|---|
| [`PretrainedConfig`] | A file that specifies a models attributes such as the number of attention heads or vocabulary size. |
| [`PreTrainedModel`] | A model (or architecture) defined by the attributes from the configuration file. For training and inference with a task, you need a model with a specific head attached to convert the raw hidden states into task-specific outputs. For example, [`PreTrainedModel`] outputs the raw hidden states but [`AutoModelForCausalLM`] adds a causal language model head on top to output the generated text. |
| Preprocessor | A class for converting raw inputs (text, images, audio, multimodal) into numerical inputs to the model. For example, [`PreTrainedTokenizer`] converts text into tensors and [`ImageProcessingMixin`] converts pixels into tensors. |

Unless you're building a custom model, you'll primarily interact with the [AutoClass](./model_doc/auto) API like [`AutoConfig`], [`AutoModelForCausalLM`], and [`AutoTokenizer`]. An `AutoClass` automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use the [`~PreTrainedModel.from_pretrained`] method to load a pretrained models weights and configuration file from the Hub into the model and preprocessor class.

<hfoptions id="base-classes">
<hfoption id="PyTorch">

When you load a model, especially a large language model (LLM), setting `device_map="auto"` automatically allocates the model weights on your device(s) beginning with the GPU.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

Tokenize the text and convert it into tensors with the tokenizer. To accelerate inference, move the model to a GPU if it's available.

```py
model_inputs = tokenizer(["Hugging Face is a"], return_tensors="pt").to("cuda")
```

The model is now ready for inference or training.

For inference, pass the tokenized inputs to the [`~GenerationMixin.generate`] API to generate text. Decode the token ids back into text with the [`~PreTrainedTokenizerBase.batch_decode`] method.

```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

</hfoption>
<hfoption id="TensorFlow">

```py
from transformers import TFAutoModelForCausalLM, AutoTokenizer

model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
```

Tokenize the text and convert it into tensors with the tokenizer. To accelerate inference, move the model to a GPU if it's available.

```py
model_inputs = tokenizer(["Hugging Face is a"], return_tensors="tf")
```

The model is now ready for inference or training.

For inference, call the [`~GenerationMixin.generate`] API to generate text and the [`~PreTrainedTokenizerBase.batch_decode`] method to convert the token ids back into text.

```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'The secret to baking a good cake is \xa0to use the right ingredients. \xa0The secret to baking a good cake is to use the right'
```

</hfoption>
</hfoptions>

For training, skip ahead to the [Trainer API](#trainer-api) section to learn how.

## Pipeline API

The [`Pipeline`] is the most convenient way to inference with a pretrained model. It supports many tasks such as text generation, image segmentation, automatic speech recognition, document question answering, and more.

> [!TIP]
> Check out the [Pipeline](./main_classes/pipelines) API reference for a complete list of available tasks.

Create a [`Pipeline`] object and select a task. By default, the [`Pipeline`] downloads and caches a default pretrained model for a given task. To choose a specific model, pass the model name to the `model` parameter.

<hfoptions id="pipeline-tasks">
<hfoption id="text generation">

Set `device="cuda"` to accelerate inference with a GPU.

```py
from transformers import pipeline

pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cuda")
```

Prompt the [`Pipeline`] with some initial text to generate more text.

```py
pipeline("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]
```

</hfoption>
<hfoption id="image segmentation">

Set `device="cuda"` to accelerate inference with a GPU.

```py
from transformers import pipeline

pipeline = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device="cuda")
```

Pass an image (a URL or local path to the image) to the [`Pipeline`].

<div class="flex justify-center">
   <img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"/>
</div>

```py
segments = pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
segments[0]["label"]
'bird'
segments[1]["label"]
'bird'
```

</hfoption>
<hfoption id="automatic speech recognition">

Set `device="cuda"` to accelerate inference with a GPU.

```py
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device="cuda")
```

Pass an audio file to the [`Pipeline`].

```py
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

</hfoption>
</hfoptions>

## Trainer API

The [`Trainer`] is an optimized training and evaluation loop for PyTorch models, a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). It abstracts away a lot of the standard boilerplate involved in manually writing a training loop, allowing you to start training faster and focus on training design choices. You only need a model, dataset, a preprocessor, and a data collator to build batches of data from the dataset.

Customize the training process with the [`TrainingArguments`] class. It provides many options for training, evaluation, and more. The training process can be as complex or simple as you want or need. Experiment with training hyperparameters and features like batch size, learning rate, mixed precision, torch.compile, and more. Or if you prefer, just use the default settings to quickly produce a baseline.

Load a model, tokenizer, and dataset for training.

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
```

Create a function to tokenize the text and convert it into tensors. Apply this function to the whole dataset with the [`datasets.Dataset.map`] method.

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

Load a data collator to create batches of data.

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Next, create an instance of [`TrainingArguments`] to customize the training process.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

Finally, pass all these separate components to [`Trainer`] and call the [`~Trainer.train`] method to start.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train()
```

Push your model to the Hub to share it with the community.

```py
trainer.push_to_hub()
```

Congratulations, you just trained your first model with Transformers!

### TensorFlow

> [!WARNING]
> Not all pretrained models are available in TensorFlow. Check which ones are implemented in [Supported models and frameworks](./index#supported-models-and-frameworks).

[`Trainer`] doesn't work with TensorFlow models, but you can still train one with [Keras](https://keras.io/). Transformers implements TensorFlow models as a standard [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), which is compatible with Keras' [compile](https://keras.io/api/models/model_training_apis/#compile-method) and [fit](https://keras.io/api/models/model_training_apis/#fit-method) methods.

Load a model, tokenizer, and dataset for training.

```py
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Create a function to tokenize the text and convert it into tensors. Apply this function to the whole dataset with the [`~datasets.Dataset.map`] method.

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])  # doctest: +SKIP
dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
```

Transformers provides the [`~TFPreTrainedModel.prepare_tf_dataset`] method to collate and batch a dataset.

```py
tf_dataset = model.prepare_tf_dataset(
    dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
)  # doctest: +SKIP
```

Finally, call [compile](https://keras.io/api/models/model_training_apis/#compile-method) to configure the model for training and [fit](https://keras.io/api/models/model_training_apis/#fit-method) to start.

```py
from tensorflow.keras.optimizers import Adam

model.compile(optimizer="adam")
model.fit(tf_dataset)  # doctest: +SKIP
```

## Next steps

Great work on completing the quickstart! You have only scratched the surface of what you can achieve with Transformers.

Now that you have a better understanding of the library, it is time to keep exploring and learning what interests you the most.

- Base classes: Learn more about the base classes and the model and processor classes that inherit from it. This will help you understand how to write your own custom models, preprocess different types of inputs (audio, images, multimodal), and how to share your model.
- Inference: Dive deeper into inference with the [`Pipeline`] API, inference with LLMs, chatting with LLMs, agents, and how to optimize inference with your machine learning framework and hardware.
- Training: Explore the [`Trainer`] API in more detail, distributed training, and optimizing training on your hardware.
- Quantization: Reduce memory and storage requirements with quantization and speed up inference by representing weights in fewer bits.
- Resources: Looking for end-to-end recipes for how to train and inference with a model for a specific task? Check out the task recipes!
