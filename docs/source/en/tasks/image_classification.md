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

# Image classification

[[open-in-colab]]

<Youtube id="tjAIM7BOYhw"/>

Image classification assigns a label or class to an image. Unlike text or audio classification, the inputs are the
pixel values that comprise an image. There are many applications for image classification, such as detecting damage
after a natural disaster, monitoring crop health, or helping screen medical images for signs of disease.

This guide illustrates how to:

1. Fine-tune [ViT](../model_doc/vit) on the [Food-101](https://huggingface.co/datasets/food101) dataset to classify a food item in an image.
2. Use your fine-tuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/image-classification)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn
```

We encourage you to log in to your Hugging Face account to upload and share your model with the community. When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load Food-101 dataset

Start by loading a smaller subset of the Food-101 dataset from the 🤗 Datasets library. This will give you a chance to
experiment and make sure everything works before spending more time training on the full dataset.

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

Split the dataset's `train` split into a train and test set with the [`~datasets.Dataset.train_test_split`] method:

```py
>>> food = food.train_test_split(test_size=0.2)
```

Then take a look at an example:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

Each example in the dataset has two fields:

- `image`: a PIL image of the food item
- `label`: the label class of the food item

To make it easier for the model to get the label name from the label id, create a dictionary that maps the label name
to an integer and vice versa:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

Now you can convert the label id to a label name:

```py
>>> id2label[str(79)]
'prime_rib'
```

## Preprocess

The next step is to load a ViT image processor to process the image into a tensor:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

<frameworkcontent>
<pt>
Apply some image transformations to the images to make the model more robust against overfitting. Here you'll use torchvision's [`transforms`](https://pytorch.org/vision/stable/transforms.html) module, but you can also use any image library you like.

Crop a random part of the image, resize it, and normalize it with the image mean and standard deviation:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

Then create a preprocessing function to apply the transforms and return the `pixel_values` - the inputs to the model - of the image:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [`~datasets.Dataset.with_transform`] method. The transforms are applied on the fly when you load an element of the dataset:

```py
>>> food = food.with_transform(transforms)
```

Now create a batch of examples using [`DefaultDataCollator`]. Unlike other data collators in 🤗 Transformers, the `DefaultDataCollator` does not apply additional preprocessing such as padding.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

To avoid overfitting and to make the model more robust, add some data augmentation to the training part of the dataset.
Here we use Keras preprocessing layers to define the transformations for the training data (includes data augmentation),
and transformations for the validation data (only center cropping, resizing and normalizing). You can use `tf.image`or
any other library you prefer.

```py
>>> from tensorflow import keras
>>> from tensorflow.keras import layers

>>> size = (image_processor.size["height"], image_processor.size["width"])

>>> train_data_augmentation = keras.Sequential(
...     [
...         layers.RandomCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...         layers.RandomFlip("horizontal"),
...         layers.RandomRotation(factor=0.02),
...         layers.RandomZoom(height_factor=0.2, width_factor=0.2),
...     ],
...     name="train_data_augmentation",
... )

>>> val_data_augmentation = keras.Sequential(
...     [
...         layers.CenterCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...     ],
...     name="val_data_augmentation",
... )
```

Next, create functions to apply appropriate transformations to a batch of images, instead of one image at a time.

```py
>>> import numpy as np
>>> import tensorflow as tf
>>> from PIL import Image


>>> def convert_to_tf_tensor(image: Image):
...     np_image = np.array(image)
...     tf_image = tf.convert_to_tensor(np_image)
...     # `expand_dims()` is used to add a batch dimension since
...     # the TF augmentation layers operates on batched inputs.
...     return tf.expand_dims(tf_image, 0)


>>> def preprocess_train(example_batch):
...     """Apply train_transforms across a batch."""
...     images = [
...         train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch


... def preprocess_val(example_batch):
...     """Apply val_transforms across a batch."""
...     images = [
...         val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch
```

Use 🤗 Datasets [`~datasets.Dataset.set_transform`] to apply the transformations on the fly:

```py
food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)
```

As a final preprocessing step, create a batch of examples using `DefaultDataCollator`. Unlike other data collators in 🤗 Transformers, the
`DefaultDataCollator` does not apply additional preprocessing, such as padding.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## Evaluate

Including a metric during training is often helpful for evaluating your model's performance. You can quickly load an
evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load
the [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

Then create a function that passes your predictions and labels to [`~evaluate.EvaluationModule.compute`] to calculate the accuracy:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

Your `compute_metrics` function is ready to go now, and you'll return to it when you set up your training.

## Train

<frameworkcontent>
<pt>
<Tip>

If you aren't familiar with finetuning a model with the [`Trainer`], take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

</Tip>

You're ready to start training your model now! Load ViT with [`AutoModelForImageClassification`]. Specify the number of labels along with the number of expected labels, and the label mappings:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

At this point, only three steps remain:

1. Define your training hyperparameters in [`TrainingArguments`]. It is important you don't remove unused columns because that'll drop the `image` column. Without the `image` column, you can't create `pixel_values`. Set `remove_unused_columns=False` to prevent this behavior! The only other required parameter is `output_dir` which specifies where to save your model. You'll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [`Trainer`] will evaluate the accuracy and save the training checkpoint.
2. Pass the training arguments to [`Trainer`] along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3. Call [`~Trainer.train`] to finetune your model.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [`~transformers.Trainer.push_to_hub`] method so everyone can use your model:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

<Tip>

If you are unfamiliar with fine-tuning a model with Keras, check out the [basic tutorial](./training#train-a-tensorflow-model-with-keras) first!

</Tip>

To fine-tune a model in TensorFlow, follow these steps:
1. Define the training hyperparameters, and set up an optimizer and a learning rate schedule.
2. Instantiate a pre-trained model.
3. Convert a 🤗 Dataset to a `tf.data.Dataset`.
4. Compile your model.
5. Add callbacks and use the `fit()` method to run the training.
6. Upload your model to 🤗 Hub to share with the community.

Start by defining the hyperparameters, optimizer and learning rate schedule:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 5
>>> num_train_steps = len(food["train"]) * num_epochs
>>> learning_rate = 3e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

Then, load ViT with [`TFAutoModelForImageClassification`] along with the label mappings:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
```

Convert your datasets to the `tf.data.Dataset` format using the [`~datasets.Dataset.to_tf_dataset`] and your `data_collator`:

```py
>>> # converting our train dataset to tf.data.Dataset
>>> tf_train_dataset = food["train"].to_tf_dataset(
...     columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
... )

>>> # converting our test dataset to tf.data.Dataset
>>> tf_eval_dataset = food["test"].to_tf_dataset(
...     columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
... )
```

Configure the model for training with `compile()`:

```py
>>> from tensorflow.keras.losses import SparseCategoricalCrossentropy

>>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
>>> model.compile(optimizer=optimizer, loss=loss)
```

To compute the accuracy from the predictions and push your model to the 🤗 Hub, use [Keras callbacks](../main_classes/keras_callbacks).
Pass your `compute_metrics` function to [KerasMetricCallback](../main_classes/keras_callbacks#transformers.KerasMetricCallback),
and use the [PushToHubCallback](../main_classes/keras_callbacks#transformers.PushToHubCallback) to upload the model:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="food_classifier",
...     tokenizer=image_processor,
...     save_strategy="no",
... )
>>> callbacks = [metric_callback, push_to_hub_callback]
```

Finally, you are ready to train your model! Call `fit()` with your training and validation datasets, the number of epochs,
and your callbacks to fine-tune the model:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
Epoch 1/5
250/250 [==============================] - 313s 1s/step - loss: 2.5623 - val_loss: 1.4161 - accuracy: 0.9290
Epoch 2/5
250/250 [==============================] - 265s 1s/step - loss: 0.9181 - val_loss: 0.6808 - accuracy: 0.9690
Epoch 3/5
250/250 [==============================] - 252s 1s/step - loss: 0.3910 - val_loss: 0.4303 - accuracy: 0.9820
Epoch 4/5
250/250 [==============================] - 251s 1s/step - loss: 0.2028 - val_loss: 0.3191 - accuracy: 0.9900
Epoch 5/5
250/250 [==============================] - 238s 949ms/step - loss: 0.1232 - val_loss: 0.3259 - accuracy: 0.9890
```

Congratulations! You have fine-tuned your model and shared it on the 🤗 Hub. You can now use it for inference!
</tf>
</frameworkcontent>


<Tip>

For a more in-depth example of how to finetune a model for image classification, take a look at the corresponding [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

</Tip>

## Inference

Great, now that you've fine-tuned a model, you can use it for inference!

Load an image you'd like to run inference on:

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="image of beignets"/>
</div>

The simplest way to try out your finetuned model for inference is to use it in a [`pipeline`]. Instantiate a `pipeline` for image classification with your model, and pass your image to it:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

You can also manually replicate the results of the `pipeline` if you'd like:

<frameworkcontent>
<pt>
Load an image processor to preprocess the image and return the `input` as PyTorch tensors:

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

Pass your inputs to the model and return the logits:

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Get the predicted label with the highest probability, and use the model's `id2label` mapping to convert it to a label:

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
Load an image processor to preprocess the image and return the `input` as TensorFlow tensors:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
>>> inputs = image_processor(image, return_tensors="tf")
```

Pass your inputs to the model and return the logits:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
>>> logits = model(**inputs).logits
```

Get the predicted label with the highest probability, and use the model's `id2label` mapping to convert it to a label:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'beignets'
```

</tf>
</frameworkcontent>
