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

Transformers is designed to be fast and easy to use so that everyone can start learning or building with transformer models.

The number of user-facing abstractions is limited to only three classes for instantiating a model, and two APIs for inference or training. This quickstart introduces you to Transformers' key features and shows you how to:

- load a pretrained model
- run inference with [`Pipeline`]
- fine-tune a model with [`Trainer`]

## Set up

To start, we recommend creating a Hugging Face [account](https://hf.co/join). An account lets you host and access version controlled models, datasets, and [Spaces](https://hf.co/spaces) on the Hugging Face [Hub](https://hf.co/docs/hub/index), a collaborative platform for discovery and building.

Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) and log in to your account.

```py
from huggingface_hub import notebook_login

notebook_login()
```

Install a machine learning framework.

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

Then install an up-to-date version of Transformers and some additional libraries from the Hugging Face ecosystem for accessing datasets and vision models, evaluating training, and optimizing training for large models.

```bash
!pip install -U transformers datasets evaluate accelerate timm
```

## Pretrained models

Each pretrained model inherits from three base classes.

| **Class** | **Description** |
|---|---|
| [`PretrainedConfig`] | A file that specifies a models attributes such as the number of attention heads or vocabulary size. |
| [`PreTrainedModel`] | A model (or architecture) defined by the model attributes from the configuration file. A pretrained model only returns the raw hidden states. For a specific task, use the appropriate model head to convert the raw hidden states into a meaningful result (for example, [`LlamaModel`] versus [`LlamaForCausalLM`]). |
| Preprocessor | A class for converting raw inputs (text, images, audio, multimodal) into numerical inputs to the model. For example, [`PreTrainedTokenizer`] converts text into tensors and [`ImageProcessingMixin`] converts pixels into tensors. |

We recommend using the [AutoClass](./model_doc/auto) API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use [`~PreTrainedModel.from_pretrained`] to load the weights and configuration file from the Hub into the model and preprocessor class.

<hfoptions id="base-classes">
<hfoption id="PyTorch">

When you load a model, configure the following parameters to ensure the model is optimally loaded.

- `device_map="auto"` automatically allocates the model weights to your fastest device first, which is typically the GPU.
- `torch_dtype="auto"` directly initializes the model weights in the data type they're stored in, which can help avoid loading the weights twice (PyTorch loads weights in torch.float32 by default).

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

Tokenize the text and return PyTorch tensors with the tokenizer. Move the model to a GPU if it's available to accelerate inference.

```py
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")
```

The model is now ready for inference or training.

For inference, pass the tokenized inputs to [`~GenerationMixin.generate`] to generate text. Decode the token ids back into text with [`~PreTrainedTokenizerBase.batch_decode`].

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

Tokenize the text and return TensorFlow tensors with the tokenizer.

```py
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="tf")
```

The model is now ready for inference or training.

For inference, pass the tokenized inputs to [`~GenerationMixin.generate`] to generate text. Decode the token ids back into text with [`~PreTrainedTokenizerBase.batch_decode`].

```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'The secret to baking a good cake is \xa0to use the right ingredients. \xa0The secret to baking a good cake is to use the right'
```

</hfoption>
</hfoptions>

> [!TIP]
> Skip ahead to the [Trainer](#trainer-api) section to learn how to fine-tune a model.

## Pipeline

The [`Pipeline`] class is the most convenient way to inference with a pretrained model. It supports many tasks such as text generation, image segmentation, automatic speech recognition, document question answering, and more.

> [!TIP]
> Refer to the [Pipeline](./main_classes/pipelines) API reference for a complete list of available tasks.

Create a [`Pipeline`] object and select a task. By default, [`Pipeline`] downloads and caches a default pretrained model for a given task. Pass the model name to the `model` parameter to choose a specific model.

<hfoptions id="pipeline-tasks">
<hfoption id="text generation">

Set `device="cuda"` to accelerate inference with a GPU.

```py
from transformers import pipeline

pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cuda")
```

Prompt [`Pipeline`] with some initial text to generate more text.

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

Pass an image - a URL or local path to the image - to [`Pipeline`].

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

Pass an audio file to [`Pipeline`].

```py
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

</hfoption>
</hfoptions>

## Trainer

[`Trainer`] is a complete training and evaluation loop for PyTorch models. It abstracts away a lot of the boilerplate usually involved in manually writing a training loop, so you can start training faster and focus on training design choices. You only need a model, dataset, a preprocessor, and a data collator to build batches of data from the dataset.

Use the [`TrainingArguments`] class to customize the training process. It provides many options for training, evaluation, and more. Experiment with training hyperparameters and features like batch size, learning rate, mixed precision, torch.compile, and more to meet your training needs. You could also use the default training parameters to quickly produce a baseline.

Load a model, tokenizer, and dataset for training.

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```

Create a function to tokenize the text and convert it into PyTorch tensors. Apply this function to the whole dataset with the [`~datasets.Dataset.map`] method.

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

Load a data collator to create batches of data and pass the tokenizer to it.

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Next, set up [`TrainingArguments`] with the training features and hyperparameters.

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

Finally, pass all these separate components to [`Trainer`] and call [`~Trainer.train`] to start.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

Share your model and tokenizer to the Hub with [`~Trainer.push_to_hub`].

```py
trainer.push_to_hub()
```

Congratulations, you just trained your first model with Transformers!

### TensorFlow

> [!WARNING]
> Not all pretrained models are available in TensorFlow. Refer to a models API doc to check whether a TensorFlow implementation is supported.

[`Trainer`] doesn't work with TensorFlow models, but you can still train a Transformers model implemented in TensorFlow with [Keras](https://keras.io/). Transformers TensorFlow models are a standard [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), which is compatible with Keras' [compile](https://keras.io/api/models/model_training_apis/#compile-method) and [fit](https://keras.io/api/models/model_training_apis/#fit-method) methods.

Load a model, tokenizer, and dataset for training.

```py
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Create a function to tokenize the text and convert it into TensorFlow tensors. Apply this function to the whole dataset with the [`~datasets.Dataset.map`] method.

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset)
```

Transformers provides the [`~TFPreTrainedModel.prepare_tf_dataset`] method to collate and batch a dataset.

```py
tf_dataset = model.prepare_tf_dataset(
    dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
)
```

Finally, call [compile](https://keras.io/api/models/model_training_apis/#compile-method) to configure the model for training and [fit](https://keras.io/api/models/model_training_apis/#fit-method) to start.

```py
from tensorflow.keras.optimizers import Adam

model.compile(optimizer="adam")
model.fit(tf_dataset)
```

## Next steps

Now that you have a better understanding of Transformers and what it offers, it's time to keep exploring and learning what interests you the most.

- **Base classes**: Learn more about the configuration, model and processor classes. This will help you understand how to create and customize models, preprocess different types of inputs (audio, images, multimodal), and how to share your model.
- **Inference**: Explore the [`Pipeline`] further, inference and chatting with LLMs, agents, and how to optimize inference with your machine learning framework and hardware.
- **Training**: Study the [`Trainer`] in more detail, as well as distributed training and optimizing training on specific hardware.
- **Quantization**: Reduce memory and storage requirements with quantization and speed up inference by representing weights with fewer bits.
- **Resources**: Looking for end-to-end recipes for how to train and inference with a model for a specific task? Check out the task recipes!
