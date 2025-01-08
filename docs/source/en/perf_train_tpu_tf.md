<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TPU

TPU, Tensor Processing Unit, is a type of hardware designed to accelerate tensor computations for training and inference. TPUs are generally accessed through Google's cloud services, but smaller TPUs are also available for free from [Google Colab](https://colab.research.google.com/notebooks/tpu.ipynb) or [Kaggle](https://www.kaggle.com/docs/tpu).

This guide focuses on training a Keras model for sequence classification on a TPU from Google Colab. Make sure the TPU runtime is enabled by going to **Runtime > Change runtime type** and selecting a TPU.

Run the command below to install the latest version of Transformers and [Datasets](https://huggingface.co/docs/datasets).

```py
!pip install --U transformers datasets
```

Create an instance of [tf.distribute.cluster_resolver.TPUClusterResolver](https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/TPUClusterResolver), and then connect to the remote cluster and initialize the TPUs.

```py
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
```

There are various distribution strategies for running your model on multiple TPUs. The [tpu.distribute.TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy) offers synchronized distributed training.

```py
strategy = tf.distribute.TPUStrategy(resolver)
```

Load and tokenize a dataset - this example uses [CoLA](https://huggingface.co/datasets/nyu-mll/glue/viewer/cola) from the GLUE benchmark - and pad all samples to the maximum length so it is easier to load as an array and to avoid XLA compilation issues.

```py
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

dataset = load_dataset("glue", "cola")["train"]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

train_data = tokenizer(
    dataset["sentence"],
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="np",
)
train_data = dict(train_data)
train_labels = np.array(dataset["label"])
```

The model **must** be created inside [Strategy.scope](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy#scope) in order to replicate the model layers on each TPU device.

```py
from transformers import TFAutoModelForSequenceClassification

with strategy.scope():
    model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    model.compile(optimizer="adam")
```

TPUs only accept [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) inputs unlike the Keras [fit](https://keras.io/api/models/model_training_apis/#fit-method) method which accepts a broader range of inputs.

```py
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

tf_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
tf_dataset = tf_dataset.shuffle(len(tf_dataset))
tf_dataset = tf_dataset.batch(BATCH_SIZE, drop_remainder=True)
```

Finally, call [fit](https://keras.io/api/models/model_training_apis/#fit-method) to start training.

```py
model.fit(tf_dataset)
```

## Large datasets

The dataset created above pads every sample to the maximum length and loads the whole dataset into memory. This may not be possible if you're working with larger datasets. When training on large datasets, you may want to create a [tf.TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) instead of stream the data.

### tf.TFRecord

[tf.TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) is the standard [tf.data](https://www.tensorflow.org/guide/data) format for storing training data. For very large training jobs, it's worth preprocessing your data and storing it in the `tf.TFRecord` format and building a `tf.data` pipeline on top. Refer to the table below to help you decide whether `tf.TFRecord` is helpful for you.

| pros | cons |
|---|---|
| works on all TPU instances | costs associated with cloud storage |
| supports huge datasets and massive throughput | some data types (images) can take a lot of space to store |
| suitable for training on entire TPU pods |  |
| preprocessing is done in advance, maximizing training speed |  |

Preprocess and tokenize the dataset before writing it to a `tf.TFRecord` to avoid writing every time the data is loaded.

An exception is made for *train-time augmentations*, because augmentations applied after writing to a `tf.TFRecord` results in the same augmentation for each epoch. Instead, apply augmentations in the `tf.data` pipeline that loads the data.

> [!TIP]
> In practice, you probably won't be able to load the entire dataset in memory. Load a chunk of the dataset at a time and convert it to `TFRecord`, and repeat until the entire dataset is in the `TFRecord` format. Then you can use a list of all the files to create a `TFRecordDataset`. The example below demonstrates a single file for simplicity.

```py
tokenized_data = tokenizer(
    dataset["sentence"],
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="np",
)
labels = dataset["label"]

with tf.io.TFRecordWriter("dataset.tfrecords") as file_writer:
    for i in range(len(labels)):
        features = {
            "input_ids": tf.train.Feature(
                int64_list=tf.train.Int64List(value=tokenized_data["input_ids"][i])
            ),
            "attention_mask": tf.train.Feature(
                int64_list=tf.train.Int64List(value=tokenized_data["attention_mask"][i])
            ),
            "labels": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[labels[i]])
            ),
        }
        features = tf.train.Features(feature=features)
        example = tf.train.Example(features=features)
        record_bytes = example.SerializeToString()
        file_writer.write(record_bytes)
```

Build a [TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) using the saved filename to load it.

```py
def decode_fn(sample):
    features = {
        "input_ids": tf.io.FixedLenFeature((128,), dtype=tf.int64),
        "attention_mask": tf.io.FixedLenFeature((128,), dtype=tf.int64),
        "labels": tf.io.FixedLenFeature((1,), dtype=tf.int64),
    }
    return tf.io.parse_example(sample, features)

# TFRecordDataset can handle gs:// paths
tf_dataset = tf.data.TFRecordDataset(["gs://matt-tf-tpu-tutorial-datasets/cola/dataset.tfrecords"])
tf_dataset = tf_dataset.map(decode_fn)
tf_dataset = tf_dataset.shuffle(len(dataset)).batch(BATCH_SIZE, drop_remainder=True)
tf_dataset = tf_dataset.apply(
    tf.data.experimental.assert_cardinality(len(labels) // BATCH_SIZE)
)
```

The dataset can now be passed to the [fit](https://keras.io/api/models/model_training_apis/#fit-method) method.

```py
model.fit(tf_dataset)
```

### Stream from raw data

Data can be stored in its native format and preprocessed in a [tf.data](https://www.tensorflow.org/guide/data) pipeline as the data is loaded. This approach isn't supported for many models with complex tokenization schemes, but some models like BERT are supported because their tokenization can be compiled. Refer to the table below to help you decide whether this approach is helpful for you.

| pros | cons |
|---|---|
| suitable for highly compressed big data in native format (images, audio) | requires writing a full preprocessing pipeline |
| convenient if raw data is available in a public cloud bucket | complex preprocessing on-the-fly can hurt throughput |
| works on all TPU instances if data is stored in Google Cloud | must place data in cloud storage if not already there |
|  | not as suitable for text data because writing a tokenization pipeline is hard (use `TFRecord` for text) |

The example below demonstrates streaming data for an image model.

Load an image dataset and get a list of the underlying image file paths and labels.

```py
from datasets import load_dataset

image_dataset = load_dataset("beans", split="train")
filenames = image_dataset["image_file_path"]
labels = image_dataset["labels"]
```

Convert the local filenames in the dataset into `gs://` paths in Google Cloud Storage.

```py
# strip everything but the category directory and filenames
base_filenames = ['/'.join(filename.split('/')[-2:]) for filename in filenames]
# prepend the Google Cloud base path to everything instead
gs_paths = ["gs://matt-tf-tpu-tutorial-datasets/beans/"+filename for filename in base_filenames]

# create tf_dataset
tf_dataset = tf.data.Dataset.from_tensor_slices(
    {"filename": gs_paths, "labels": labels}
)
tf_dataset = tf_dataset.shuffle(len(tf_dataset))
```

Transformers preprocessing classes like [`AutoImageProcessor`] are framework-agnostic and can't be compiled into a pipeline by `tf.data`. To get around this, get the normalization values (`mean` and `std`) from the [`AutoImageProcessor`] and use them in the `tf.data` pipeline.

```py
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
image_size = (processor.size["height"], processor.size["width"])
image_mean = processor.image_mean
image_std = processor.image_std
```

Use these normalization values to create a function to load and preprocess the images.

```py
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

def decode_fn(sample):
    image_data = tf.io.read_file(sample["filename"])
    image = tf.io.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, image_size)
    array = tf.cast(image, tf.float32)
    array /= 255.0
    array = (array - image_mean) / image_std
    array = tf.transpose(array, perm=[2, 0, 1])
    return {"pixel_values": array, "labels": sample["labels"]}

tf_dataset = tf_dataset.map(decode_fn)
tf_dataset = tf_dataset.batch(BATCH_SIZE, drop_remainder=True)
print(tf_dataset.element_spec)
```

The dataset can now be passed to the [fit](https://keras.io/api/models/model_training_apis/#fit-method) method.

```py
from transformers import TFAutoModelForImageClassification

with strategy.scope():
    model = TFAutoModelForImageClassification.from_pretrained(image_model_checkpoint)
    model.compile(optimizer="adam")

model.fit(tf_dataset)
```

### Stream dataset with prepare_tf_dataset

[`~TFPreTrainedModel.prepare_tf_dataset`] creates a `tf.data` pipeline that loads samples from [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). The pipeline uses [tf.numpy_function]() or [`~datasets.Dataset.from_generator`], which can't be compiled by TensorFlow, to access the underlying `tf.data.Dataset`. It also won't work on a Colab TPU or TPU Nodes because the pipeline stream data from a local disk. Refer to the table below to help you decide whether this approach is helpful for you.

| pros | cons |  |  |  |
|---|---|---|---|---|
| simple code | only works on TPU VM |  |  |  |
| same approach on TPU/GPU | data must be available as a Hugging Face Dataset |  |  |  |
| dataset doesn't have to fit in memory | data must fit on local storage |  |  |  |
| supports variable padding | data loading may be a bottleneck on a big TPU pod slice |  |  |  |

[`~TFPreTrainedModel.prepare_tf_dataset`] only works on **TPU VM**. Add the tokenizer output as columns in the dataset since the dataset is stored on disk, which means it can handle data larger than the available memory. Use [`~TFPreTrainedModel.prepare_tf_dataset`] to stream data from the dataset by wrapping it with a `tf.data` pipeline.

```py
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"], padding="max_length", truncation=True, max_length=128
    )
# add the tokenizer output to the dataset as new columns
dataset = dataset.map(tokenize_function)

# prepare_tf_dataset() chooses columns that match the models input names
tf_dataset = model.prepare_tf_dataset(
    dataset, batch_size=BATCH_SIZE, shuffle=True, tokenizer=tokenizer
)
```

The dataset can now be passed to the [fit](https://keras.io/api/models/model_training_apis/#fit-method) method.

```py
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

with strategy.scope():
    model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    model.compile(optimizer="adam")

model.fit(tf_dataset)
```

## TPU types

There are two types of TPUs, a TPU Node and a TPU VM.

A TPU Node indirectly accesses a remote TPU. It requires a separate VM to initialize your network and data pipeline and then forwards it to the remote node. Google Colab TPUs are an example of a TPU Node. You can't use local data because the TPU is remotely located, and data must be stored in Google Cloud Storage where the data pipeline can access it.

TPU VM are connected directly to the machine the TPU is located on, and they are generally easier to work with, especially when it comes to your data pipeline.

> [!TIP]
> We recommend avoiding TPU Nodes if possible because it is more difficult to debug than TPU VMs. TPU Nodes may also be unsupported in the future and become a legacy access method.

A single TPU (v2-8, v3-8, v4-8) runs 8 replicas. TPUs can exist in **pods** which run hundreds or even thousands of replicas simultaneously. When you only use a portion of a pod, it is referred to as a **pod slice**. On Google Colab, you'll typically get a single v2-8 TPU.

## XLA

[XLA](https://openxla.org/xla) is a linear algebra compiler for high-performance execution and it is used by default to improve performance on TPUs.

Before executing your code on a TPU, it's a good idea to try it first on a CPU or GPU because it is easier to debug. You can train for a few steps to make sure the model and data pipeline work as expected. Set `jit_compile=True` in the [compile](https://keras.io/api/models/model_training_apis/#compile-method) method to enable XLA compilation (but remember to remove this line of code before running on a TPU).

The section below outlines three rules for making your code XLA-compatible. Transformers enforce the first two rules for models and loss functions by default, but don't forget about them if you're writing your own models and loss functions.

### Data dependent conditionals

Any `if` statements cannot depend on values inside a [tf.Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor). The code below can't be compiled by XLA.

```py
if tf.reduce_sum(tensor) > 10:
    tensor = tensor / 2.0
```

To compile with XLA, use [tf.cond](https://www.tensorflow.org/api_docs/python/tf/cond) or remove the conditional and use indicator variables instead as shown below.

```py
sum_over_10 = tf.cast(tf.reduce_sum(tensor) > 10, tf.float32)
tensor = tensor / (1.0 + sum_over_10)
```

### Data dependent shapes

The shape of a [tf.Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor) cannot depend on their values. For example, [tf.unique](https://www.tensorflow.org/api_docs/python/tf/unique) can't be compiled because it returns a tensor containing an instance of each unique value in the input. The shape of this output depends on how repetitive the input [tf.Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor) is.

This is an issue during **label masking**, where labels are set to a negative value to indicate they should be ignored when computing the loss. The code below can't be compiled by XLA because the shape of `masked_outputs` and `masked_labels` depend on how many positions are masked.

```py
label_mask = labels >= 0
masked_outputs = outputs[label_mask]
masked_labels = labels[label_mask]
loss = compute_loss(masked_outputs, masked_labels)
mean_loss = torch.mean(loss)
```

To compile with XLA, avoid the data-dependent shapes by computing the loss for every position and zeroing out the masked positions in both the numerator and denominator when calculating the mean. Convert `tf.bool` to `tf.float32` as an indicator variable to make your code XLA-compatible.

```py
label_mask = tf.cast(labels >= 0, tf.float32)
loss = compute_loss(outputs, labels)
loss = loss * label_mask
mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(label_mask)
```

### Recompile different input shapes

XLA recompiles your model if input shapes are variable which create huge performance problems. It is especially common in text models because input texts have variable lengths after tokenization.

> [!WARNING]
> Execessive padding can also severely slow down training because requires more compute and memory to process.

To avoid different shapes, use padding to pad all your inputs to the same length and use an `attention_mask`. Try padding batches of samples to a multiple of 32 or 64 tokens. Use the parameters `padding="max_length"`, `padding="longest"`, or `pad_to_multiple_of` to help with padding. This often increases the number of tokens by a small amount, but it significantly reduces the number of unique input shapes because every input shape is a multiple of 32 or 64. Fewer unique input shapes requires fewer recompilation.