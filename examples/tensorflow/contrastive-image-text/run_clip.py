#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Team All rights reserved.
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
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf
from datasets import load_dataset
from PIL import Image

import transformers
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
    PushToHubCallback,
    TFAutoModel,
    TFTrainingArguments,
    TFVisionTextDualEncoderModel,
    create_optimizer,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version(
    "datasets>=1.8.0", "To fix: pip install -r examples/tensorflow/contrastive-image-text/requirements.txt"
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}, default=None
    )
    vision_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained image model or model identifier from huggingface.co/models"},
        default=None,
    )
    text_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained text model or model identifier from huggingface.co/models"}, default=None
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
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
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


def crop_to_square(image):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    if height > width:
        image = tf.image.crop_to_bounding_box(image, (height - width) // 2, 0, width, width)
    elif width > height:
        image = tf.image.crop_to_bounding_box(image, 0, (width - height) // 2, height, height)
    return image


def load_as_tf_dataset(dataset, image_column, image_size, mean, std, batch_size, shuffle):
    dataset = dataset.with_format("tensorflow")[:]  # Load the dataset as tensor slices, but not the images yet!
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

    def load_image(sample):
        image_path = sample[image_column]
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = crop_to_square(image)
        image = tf.image.resize(image, [image_size, image_size], method="bicubic", antialias=True)
        image = image / 255.0
        image = (image - mean) / std
        image = tf.transpose(image, perm=[2, 0, 1])  # Convert to channels-first
        sample["pixel_values"] = image
        del sample[image_column]
        return sample

    if shuffle:
        tf_dataset = tf_dataset.shuffle(len(tf_dataset))
    tf_dataset = tf_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=shuffle)
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset


def main():
    # 1. Parse input arguments
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

    if model_args.model_name_or_path is not None:
        if model_args.vision_model_name_or_path is not None or model_args.text_model_name_or_path is not None:
            raise ValueError(
                "If using model_name_or_path, you cannot specify separate image/text model paths as well!"
            )

    if model_args.vision_model_name_or_path is not None or model_args.text_model_name_or_path is not None:
        if model_args.model_name_or_path is not None:
            raise ValueError(
                "If using separate image/text model paths, you cannot specify model_name_or_path as well!"
            )
        if not (model_args.vision_model_name_or_path is not None and model_args.text_model_name_or_path is not None):
            raise ValueError(
                "If using separate image/text model paths, you must specify both vision_model_name_or_path "
                "and text_model_name_or_path!"
            )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/TensorFlow versions.
    send_example_telemetry("run_clip", model_args, data_args, framework="tensorflow")

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.text_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        # Load image_processor, in this script we only use this to get the mean and std for normalization.
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.image_processor_name or model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        with training_args.strategy.scope():
            model = TFAutoModel.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        # Load image_processor, in this script we only use this to get the mean and std for normalization.
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.image_processor_name or model_args.vision_model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        with training_args.strategy.scope():
            model = TFVisionTextDualEncoderModel.from_vision_text_pretrained(
                vision_model_name_or_path=model_args.vision_model_name_or_path,
                text_model_name_or_path=model_args.text_model_name_or_path,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    config = model.config

    if model_args.freeze_vision_model:
        model.vision_model.trainable = False

    if model_args.freeze_text_model:
        model.text_model.trainable = False

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # # 7. Preprocessing the datasets.

    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open(image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        tf_train_dataset = load_as_tf_dataset(
            dataset=train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            image_column=image_column,
            image_size=config.vision_config.image_size,
            mean=image_processor.image_mean,
            std=image_processor.image_std,
            shuffle=True,
        )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        tf_eval_dataset = load_as_tf_dataset(
            dataset=eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            image_column=image_column,
            image_size=config.vision_config.image_size,
            mean=image_processor.image_mean,
            std=image_processor.image_std,
            shuffle=False,
        )

    # 8. Preparing push_to_hub and model card
    push_to_hub_model_id = training_args.push_to_hub_model_id
    if model_args.model_name_or_path is not None:
        model_name = model_args.model_name_or_path.split("/")[-1]
    else:
        vision_name = model_args.vision_model_name_or_path.split("/")[-1]
        text_name = model_args.text_model_name_or_path.split("/")[-1]
        model_name = f"{vision_name}-{text_name}"
    if not push_to_hub_model_id:
        if data_args.dataset_name is not None:
            push_to_hub_model_id = f"{model_name}-finetuned-{data_args.dataset_name}"
        else:
            push_to_hub_model_id = f"{model_name}-finetuned-contrastive-image-text-modeling"

    model_card_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "contrastive-image-text-modeling"}
    if data_args.dataset_name is not None:
        model_card_kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            model_card_kwargs["dataset_args"] = data_args.dataset_config_name
            model_card_kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            model_card_kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        callbacks = [
            PushToHubCallback(
                output_dir=training_args.output_dir,
                hub_model_id=push_to_hub_model_id,
                hub_token=training_args.push_to_hub_token,
                tokenizer=tokenizer,
                **model_card_kwargs,
            )
        ]
    else:
        callbacks = []

    # # 9. Training
    if training_args.do_train:
        num_train_steps = int(len(tf_train_dataset) * int(training_args.num_train_epochs))
        if training_args.warmup_steps > 0:
            num_warmup_steps = training_args.warmup_steps
        elif training_args.warmup_ratio > 0:
            num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
        else:
            num_warmup_steps = 0
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
            adam_global_clipnorm=training_args.max_grad_norm,
        )
        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, jit_compile=training_args.xla)

        if not training_args.do_eval:
            tf_eval_dataset = None
        model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
            callbacks=callbacks,
        )

    # # 10. Evaluation

    if training_args.do_eval and not training_args.do_train:
        model.evaluate(tf_eval_dataset)


if __name__ == "__main__":
    main()
