#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
Fine-tuning a 🤗 Transformers model for image classification.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=image-classification
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image

import transformers
from transformers import (
    TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    DefaultDataCollator,
    HfArgumentParser,
    PushToHubCallback,
    TFAutoModelForImageClassification,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.keras_callbacks import KerasMetricCallback
from transformers.modeling_tf_utils import keras
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def center_crop(image, size):
    size = (size, size) if isinstance(size, int) else size
    orig_height, orig_width, _ = image.shape
    crop_height, crop_width = size
    top = (orig_height - orig_width) // 2
    left = (orig_width - crop_width) // 2
    image = tf.image.crop_to_bounding_box(image, top, left, crop_height, crop_width)
    return image


# Numpy and TensorFlow compatible version of PyTorch RandomResizedCrop. Code adapted from:
# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
def random_crop(image, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    height, width, _ = image.shape
    area = height * width
    log_ratio = np.log(ratio)
    for _ in range(10):
        target_area = np.random.uniform(*scale) * area
        aspect_ratio = np.exp(np.random.uniform(*log_ratio))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = np.random.randint(0, height - h + 1)
            j = np.random.randint(0, width - w + 1)
            return image[i : i + h, j : j + w, :]

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    w = width if in_ratio < min(ratio) else int(round(height * max(ratio)))
    h = height if in_ratio > max(ratio) else int(round(width / min(ratio)))
    i = (height - h) // 2
    j = (width - w) // 2
    return image[i : i + h, j : j + w, :]


def random_resized_crop(image, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    size = (size, size) if isinstance(size, int) else size
    image = random_crop(image, scale, ratio)
    image = tf.image.resize(image, size)
    return image


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/TensorFlow versions.
    send_example_telemetry("run_image_classification", model_args, data_args, framework="tensorflow")

    # Checkpoints. Find the checkpoint the use when loading the model.
    checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # region Dataset and labels
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load model image processor and configuration
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Define our data preprocessing function. It takes an image file path as input and returns
    # Write a note describing the resizing behaviour.
    if "shortest_edge" in image_processor.size:
        # We instead set the target size as (shortest_edge, shortest_edge) to here to ensure all images are batchable.
        image_size = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    else:
        image_size = (image_processor.size["height"], image_processor.size["width"])

    def _train_transforms(image):
        img_size = image_size
        image = keras.utils.img_to_array(image)
        image = random_resized_crop(image, size=img_size)
        image = tf.image.random_flip_left_right(image)
        image /= 255.0
        image = (image - image_processor.image_mean) / image_processor.image_std
        image = tf.transpose(image, perm=[2, 0, 1])
        return image

    def _val_transforms(image):
        image = keras.utils.img_to_array(image)
        image = tf.image.resize(image, size=image_size)
        # image = np.array(image) # FIXME - use tf.image function
        image = center_crop(image, size=image_size)
        image /= 255.0
        image = (image - image_processor.image_mean) / image_processor.image_std
        image = tf.transpose(image, perm=[2, 0, 1])
        return image

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    train_dataset = None
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            train_transforms,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # Set the validation transforms
        eval_dataset = eval_dataset.map(
            val_transforms,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    predict_dataset = None
    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        # Set the test transforms
        predict_dataset = predict_dataset.map(
            val_transforms,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    collate_fn = DefaultDataCollator(return_tensors="np")

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        logits, label_ids = p
        predictions = np.argmax(logits, axis=-1)
        metrics = metric.compute(predictions=predictions, references=label_ids)
        return metrics

    with training_args.strategy.scope():
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint

        model = TFAutoModelForImageClassification.from_pretrained(
            model_path,
            config=config,
            from_pt=bool(".bin" in model_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        if training_args.do_train:
            num_train_steps = int(len(train_dataset) * training_args.num_train_epochs)
            if training_args.warmup_steps > 0:
                num_warmpup_steps = int(training_args.warmup_steps)
            elif training_args.warmup_ratio > 0:
                num_warmpup_steps = int(training_args.warmup_ratio * num_train_steps)
            else:
                num_warmpup_steps = 0

            optimizer, _ = create_optimizer(
                init_lr=training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmpup_steps,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
                weight_decay_rate=training_args.weight_decay,
                adam_global_clipnorm=training_args.max_grad_norm,
            )
            # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
            # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
            # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
            # yourself if you use this method, whereas they are automatically inferred from the model input names when
            # using model.prepare_tf_dataset()
            # For more info see the docs:
            # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
            # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset
            train_dataset = model.prepare_tf_dataset(
                train_dataset,
                shuffle=True,
                batch_size=total_train_batch_size,
                collate_fn=collate_fn,
            ).with_options(dataset_options)
        else:
            optimizer = "sgd"  # Just write anything because we won't be using it

        if training_args.do_eval:
            eval_dataset = model.prepare_tf_dataset(
                eval_dataset,
                shuffle=False,
                batch_size=total_eval_batch_size,
                collate_fn=collate_fn,
            ).with_options(dataset_options)

        if training_args.do_predict:
            predict_dataset = model.prepare_tf_dataset(
                predict_dataset,
                shuffle=False,
                batch_size=total_eval_batch_size,
                collate_fn=collate_fn,
            ).with_options(dataset_options)

        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, jit_compile=training_args.xla, metrics=["accuracy"])

        push_to_hub_model_id = training_args.push_to_hub_model_id
        if not push_to_hub_model_id:
            model_name = model_args.model_name_or_path.split("/")[-1]
            push_to_hub_model_id = f"{model_name}-finetuned-image-classification"

        model_card_kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "image-classification",
            "dataset": data_args.dataset_name,
            "tags": ["image-classification", "tensorflow", "vision"],
        }

        callbacks = []
        if eval_dataset is not None:
            callbacks.append(KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=eval_dataset))
        if training_args.push_to_hub:
            callbacks.append(
                PushToHubCallback(
                    output_dir=training_args.output_dir,
                    hub_model_id=push_to_hub_model_id,
                    hub_token=training_args.push_to_hub_token,
                    tokenizer=image_processor,
                    **model_card_kwargs,
                )
            )

        if training_args.do_train:
            model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )

        if training_args.do_eval:
            n_eval_batches = len(eval_dataset)
            eval_predictions = model.predict(eval_dataset, steps=n_eval_batches)
            eval_labels = dataset["validation"]["labels"][: n_eval_batches * total_eval_batch_size]
            eval_metrics = compute_metrics((eval_predictions.logits, eval_labels))
            logging.info("Eval metrics:")
            for metric_name, value in eval_metrics.items():
                logging.info(f"{metric_name}: {value:.3f}")

        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
                f.write(json.dumps(eval_metrics))

        if training_args.do_predict:
            n_predict_batches = len(predict_dataset)
            test_predictions = model.predict(predict_dataset, steps=n_predict_batches)
            test_labels = dataset["validation"]["labels"][: n_predict_batches * total_eval_batch_size]
            test_metrics = compute_metrics((test_predictions.logits, test_labels))
            logging.info("Test metrics:")
            for metric_name, value in test_metrics.items():
                logging.info(f"{metric_name}: {value:.3f}")

        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
