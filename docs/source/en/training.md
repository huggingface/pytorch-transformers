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

# Fine-tune a pretrained model

[[open-in-colab]]

There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:

* Fine-tune a pretrained model with 🤗 Transformers [`Trainer`].
* Fine-tune a pretrained model in TensorFlow with Keras.
* Fine-tune a pretrained model in native PyTorch.

<a id='data-processing'></a>

## Prepare a dataset

<Youtube id="_BZearw7f0w"/>

Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!

Begin by loading the [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) dataset:

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process#map) method to apply a preprocessing function over the entire dataset:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Train

At this point, you should follow the section corresponding to the framework you want to use. You can use the links
in the right sidebar to jump to the one you want - and if you want to hide all of the content for a given framework,
just use the button at the top-right of that framework's block!

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

## Train with PyTorch Trainer

🤗 Transformers provides a [`Trainer`] class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [`Trainer`] API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

Start by loading your model and specify the number of expected labels. From the Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields), you know there are five labels.

By default, the weights are loaded in full precision (torch.float32) regardless of the actual data type the weights are stored in such as torch.float16. Set `torch_dtype="auto"` to load the weights in the data type defined in a model's `config.json` file to automatically load the most memory-optimal data type.

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
```

<Tip>

You will see a warning about some of the pretrained weights not being used and some weights being randomly
initialized. Don't worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.

</Tip>

### Training hyperparameters

Next, create a [`TrainingArguments`] class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training [hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments), but feel free to experiment with these to find your optimal settings.

Specify where to save the checkpoints from your training:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Evaluate

[`Trainer`] does not automatically evaluate model performance during training. You'll need to pass [`Trainer`] a function to compute and report metrics. The [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) library provides a simple [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) function you can load with the [`evaluate.load`] (see this [quicktour](https://huggingface.co/docs/evaluate/a_quick_tour) for more information) function:

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

Call [`~evaluate.compute`] on `metric` to calculate the accuracy of your predictions. Before passing your predictions to `compute`, you need to convert the logits to predictions (remember all 🤗 Transformers models return logits):

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

If you'd like to monitor your evaluation metrics during fine-tuning, specify the `eval_strategy` parameter in your training arguments to report the evaluation metric at the end of each epoch:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

### Trainer

Create a [`Trainer`] object with your model, training arguments, training and test datasets, and evaluation function:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Then fine-tune your model by calling [`~transformers.Trainer.train`]:

```py
>>> trainer.train()
```
</pt>
<tf>
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

## Train a TensorFlow model with Keras

You can also train 🤗 Transformers models in TensorFlow with the Keras API!

### Loading data for Keras

When you want to train a 🤗 Transformers model with the Keras API, you need to convert your dataset to a format that
Keras understands. If your dataset is small, you can just convert the whole thing to NumPy arrays and pass it to Keras.
Let's try that first before we do anything more complicated.

First, load a dataset. We'll use the CoLA dataset from the [GLUE benchmark](https://huggingface.co/datasets/glue),
since it's a simple binary text classification task, and just take the training split for now.

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # Just take the training split for now
```

Next, load a tokenizer and tokenize the data as NumPy arrays. Note that the labels are already a list of 0 and 1s,
so we can just convert that directly to a NumPy array without tokenization!

```py
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1
```

Finally, load, [`compile`](https://keras.io/api/models/model_training_apis/#compile-method), and [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) the model. Note that Transformers models all have a default task-relevant loss function, so you don't need to specify one unless you want to:

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tokenized_data, labels)
```

<Tip>

You don't have to pass a loss argument to your models when you `compile()` them! Hugging Face models automatically
choose a loss that is appropriate for their task and model architecture if this argument is left blank. You can always
override this by specifying a loss yourself if you want to!

</Tip>

This approach works great for smaller datasets, but for larger datasets, you might find it starts to become a problem. Why?
Because the tokenized array and labels would have to be fully loaded into memory, and because NumPy doesn’t handle
“jagged” arrays, so every tokenized sample would have to be padded to the length of the longest sample in the whole
dataset. That’s going to make your array even bigger, and all those padding tokens will slow down training too!

### Loading data as a tf.data.Dataset

If you want to avoid slowing down training, you can load your data as a `tf.data.Dataset` instead. Although you can write your own
`tf.data` pipeline if you want, we have two convenience methods for doing this:

- [`~TFPreTrainedModel.prepare_tf_dataset`]: This is the method we recommend in most cases. Because it is a method
on your model, it can inspect the model to automatically figure out which columns are usable as model inputs, and
discard the others to make a simpler, more performant dataset.
- [`~datasets.Dataset.to_tf_dataset`]: This method is more low-level, and is useful when you want to exactly control how
your dataset is created, by specifying exactly which `columns` and `label_cols` to include.

Before you can use [`~TFPreTrainedModel.prepare_tf_dataset`], you will need to add the tokenizer outputs to your dataset as columns, as shown in
the following code sample:

```py
def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

Remember that Hugging Face datasets are stored on disk by default, so this will not inflate your memory usage! Once the
columns have been added, you can stream batches from the dataset and add padding to each batch, which greatly
reduces the number of padding tokens compared to padding the entire dataset.


```py
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

Note that in the code sample above, you need to pass the tokenizer to `prepare_tf_dataset` so it can correctly pad batches as they're loaded.
If all the samples in your dataset are the same length and no padding is necessary, you can skip this argument.
If you need to do something more complex than just padding samples (e.g. corrupting tokens for masked language
modelling), you can use the `collate_fn` argument instead to pass a function that will be called to transform the
list of samples into a batch and apply any preprocessing you want. See our
[examples](https://github.com/huggingface/transformers/tree/main/examples) or
[notebooks](https://huggingface.co/docs/transformers/notebooks) to see this approach in action.

Once you've created a `tf.data.Dataset`, you can compile and fit the model as before:

```py
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tf_dataset)
```

</tf>
</frameworkcontent>

<a id='pytorch_native'></a>

## Train in native PyTorch

<frameworkcontent>
<pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`] takes care of the training loop and allows you to fine-tune a model in a single line of code. For users who prefer to write their own training loop, you can also fine-tune a 🤗 Transformers model in native PyTorch.

At this point, you may need to restart your notebook or execute the following code to free some memory:

```py
from accelerate.utils.memory import clear_device_cache
del model
del trainer
clear_device_cache()
```

Next, manually postprocess `tokenized_dataset` to prepare it for training.

1. Remove the `text` column because the model does not accept raw text as an input:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. Rename the `label` column to `labels` because the model expects the argument to be named `labels`:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. Set the format of the dataset to return PyTorch tensors instead of lists:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

Create a `DataLoader` for your training and test datasets so you can iterate over batches of data:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Load your model with the number of expected labels:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### Optimizer and learning rate scheduler

Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

Create the default learning rate scheduler from [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

Lastly, specify `device` to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.

```py
>>> import torch
>>> from accelerate.test_utils.testing import get_backend

>>> device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
>>> model.to(device)
```

<Tip>

Get free access to a cloud GPU if you don't have one with a hosted notebook like [Colaboratory](https://colab.research.google.com/) or [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

</Tip>

Great, now you are ready to train! 🥳 

### Training loop

To keep track of your training progress, use the [tqdm](https://tqdm.github.io/) library to add a progress bar over the number of training steps:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### Evaluate

Just like how you added an evaluation function to [`Trainer`], you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you'll accumulate all the batches with [`~evaluate.add_batch`] and calculate the metric at the very end.

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```
</pt>
</frameworkcontent>

<a id='additional-resources'></a>

## Additional resources

For more fine-tuning examples, refer to:

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) includes scripts
  to train common NLP tasks in PyTorch and TensorFlow.

- [🤗 Transformers Notebooks](notebooks) contains various notebooks on how to fine-tune a model for specific tasks in PyTorch and TensorFlow.
