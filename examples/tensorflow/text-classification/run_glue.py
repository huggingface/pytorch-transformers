#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

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

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    HfArgumentParser,
    PretrainedConfig,
    PushToHubCallback,
    TFAutoModelForSequenceClassification,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


# region Command-line arguments
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    predict_file: str = field(
        metadata={"help": "A file containing user-supplied examples to make predictions for"},
        default=None,
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
        self.task_name = self.task_name.lower()
        if self.task_name not in task_to_keys.keys():
            raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


# endregion


def main():
    # region Argument parsing
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
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args, framework="tensorflow")

    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        exit("Must specify at least one of --do_train, --do_eval or --do_predict!")
    # endregion

    # region Checkpoints
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
    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # region Dataset and labels
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub. In distributed training, the load_dataset function guarantee
    # that only one local process can concurrently download the dataset.
    datasets = load_dataset(
        "nyu-mll/glue",
        data_args.task_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    if data_args.predict_file is not None:
        logger.info("Preparing user-supplied file for predictions...")

        data_files = {"data": data_args.predict_file}

        for key in data_files.keys():
            logger.info(f"Loading a local file for {key}: {data_files[key]}")

        if data_args.predict_file.endswith(".csv"):
            # Loading a dataset from local csv files
            user_dataset = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            user_dataset = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        needed_keys = task_to_keys[data_args.task_name]
        for key in needed_keys:
            assert key in user_dataset["data"].features, f"Your supplied predict_file is missing the {key} key!"
        datasets["user_data"] = user_dataset["data"]
    # endregion

    # region Load model config and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # endregion

    # region Dataset preprocessing
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if config.label2id != PretrainedConfig(num_labels=num_labels).label2id and not is_regression:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
            label_to_id = {label: i for i, label in enumerate(label_list)}
    if label_to_id is not None:
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    if data_args.pad_to_max_length:
        data_collator = DefaultDataCollator(return_tensors="np")
    else:
        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="np")
    # endregion

    # region Metric function
    metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)

    def compute_metrics(preds, label_ids):
        preds = preds["logits"]
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # endregion

    with training_args.strategy.scope():
        # region Load pretrained model
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # endregion

        # region Convert data to a tf.data.Dataset
        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        num_replicas = training_args.strategy.num_replicas_in_sync

        tf_data = {}
        max_samples = {
            "train": data_args.max_train_samples,
            "validation": data_args.max_eval_samples,
            "validation_matched": data_args.max_eval_samples,
            "validation_mismatched": data_args.max_eval_samples,
            "test": data_args.max_predict_samples,
            "test_matched": data_args.max_predict_samples,
            "test_mismatched": data_args.max_predict_samples,
            "user_data": None,
        }
        for key in datasets.keys():
            if key == "train" or key.startswith("validation"):
                assert "label" in datasets[key].features, f"Missing labels from {key} data!"
            if key == "train":
                shuffle = True
                batch_size = training_args.per_device_train_batch_size * num_replicas
            else:
                shuffle = False
                batch_size = training_args.per_device_eval_batch_size * num_replicas
            samples_limit = max_samples[key]
            dataset = datasets[key]
            if samples_limit is not None:
                dataset = dataset.select(range(samples_limit))

            # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
            # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
            # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
            # yourself if you use this method, whereas they are automatically inferred from the model input names when
            # using model.prepare_tf_dataset()
            # For more info see the docs:
            # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
            # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset
            data = model.prepare_tf_dataset(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=data_collator,
                tokenizer=tokenizer,
            )
            data = data.with_options(dataset_options)
            tf_data[key] = data
        # endregion

        # region Optimizer, loss and compilation
        if training_args.do_train:
            num_train_steps = len(tf_data["train"]) * training_args.num_train_epochs
            if training_args.warmup_steps > 0:
                num_warmup_steps = training_args.warmup_steps
            elif training_args.warmup_ratio > 0:
                num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
            else:
                num_warmup_steps = 0

            optimizer, schedule = create_optimizer(
                init_lr=training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
                weight_decay_rate=training_args.weight_decay,
                adam_global_clipnorm=training_args.max_grad_norm,
            )
        else:
            optimizer = "sgd"  # Just write anything because we won't be using it
        if is_regression:
            metrics = []
        else:
            metrics = ["accuracy"]
        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, metrics=metrics, jit_compile=training_args.xla)
        # endregion

        # region Preparing push_to_hub and model card
        push_to_hub_model_id = training_args.push_to_hub_model_id
        model_name = model_args.model_name_or_path.split("/")[-1]
        if not push_to_hub_model_id:
            push_to_hub_model_id = f"{model_name}-finetuned-glue"

        model_card_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
        model_card_kwargs["task_name"] = data_args.task_name

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
        # endregion

        # region Training and validation
        if training_args.do_train:
            if training_args.do_eval and not data_args.task_name == "mnli":
                # Do both evaluation and training in the Keras fit loop, unless the task is MNLI
                # because MNLI has two validation sets
                validation_data = tf_data["validation"]
            else:
                validation_data = None
            model.fit(
                tf_data["train"],
                validation_data=validation_data,
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )
        # endregion

        # region Evaluation
        if training_args.do_eval:
            # We normally do validation as part of the Keras fit loop, but we run it independently
            # if there was no fit() step (because we didn't train the model) or if the task is MNLI,
            # because MNLI has a separate validation-mismatched validation set

            # In this example, we compute advanced metrics only at the end of training, and only compute
            # loss and accuracy on the validation set each epoch, but
            # if you'd like to compute metrics every epoch that are too complex to be written as
            # standard Keras metrics, you can use our KerasMetricCallback. See
            # https://huggingface.co/docs/transformers/main/en/main_classes/keras_callbacks
            logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            if data_args.task_name == "mnli":
                tasks = ["mnli", "mnli-mm"]
                tf_datasets = [tf_data["validation_matched"], tf_data["validation_mismatched"]]
                raw_datasets = [datasets["validation_matched"], datasets["validation_mismatched"]]
            else:
                tasks = [data_args.task_name]
                tf_datasets = [tf_data["validation"]]
                raw_datasets = [datasets["validation"]]

            for raw_dataset, tf_dataset, task in zip(raw_datasets, tf_datasets, tasks):
                eval_predictions = model.predict(tf_dataset)
                eval_metrics = compute_metrics(eval_predictions, raw_dataset["label"])
                print(f"Evaluation metrics ({task}):")
                print(eval_metrics)
                if training_args.output_dir is not None:
                    output_eval_file = os.path.join(training_args.output_dir, "all_results.json")
                    with open(output_eval_file, "w") as writer:
                        writer.write(json.dumps(eval_metrics))

        # endregion

        # region Prediction
        if training_args.do_predict or data_args.predict_file:
            logger.info("*** Predict ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = []
            tf_datasets = []
            raw_datasets = []
            if training_args.do_predict:
                if data_args.task_name == "mnli":
                    tasks.extend(["mnli", "mnli-mm"])
                    tf_datasets.extend([tf_data["test_matched"], tf_data["test_mismatched"]])
                    raw_datasets.extend([datasets["test_matched"], datasets["test_mismatched"]])
                else:
                    tasks.append(data_args.task_name)
                    tf_datasets.append(tf_data["test"])
                    raw_datasets.append(datasets["test"])
            if data_args.predict_file:
                tasks.append("user_data")
                tf_datasets.append(tf_data["user_data"])
                raw_datasets.append(datasets["user_data"])

            for raw_dataset, tf_dataset, task in zip(raw_datasets, tf_datasets, tasks):
                test_predictions = model.predict(tf_dataset)
                if "label" in raw_dataset:
                    test_metrics = compute_metrics(test_predictions, raw_dataset["label"])
                    print(f"Test metrics ({task}):")
                    print(test_metrics)

                if is_regression:
                    predictions_to_write = np.squeeze(test_predictions["logits"])
                else:
                    predictions_to_write = np.argmax(test_predictions["logits"], axis=1)

                output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Writing prediction results for {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions_to_write):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = model.config.id2label[item]
                            writer.write(f"{index}\t{item}\n")
        # endregion

        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
