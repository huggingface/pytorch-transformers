#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

import datasets
import tensorflow as tf
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_NAME,
    TF2_WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    DefaultDataCollator,
    HfArgumentParser,
    PushToHubCallback,
    TFAutoModelForMultipleChoice,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.utils import check_min_version, send_example_telemetry


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.49.0.dev0")

logger = logging.getLogger(__name__)


# region Arguments
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
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

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


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
    send_example_telemetry("run_swag", model_args, data_args, framework="tensorflow")

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # endregion

    # region Checkpoints
    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        if (output_dir / CONFIG_NAME).is_file() and (output_dir / TF2_WEIGHTS_NAME).is_file():
            checkpoint = output_dir
            logger.info(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )
    # endregion

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # region Load datasets
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Downloading and loading the swag dataset from the hub.
        raw_datasets = load_dataset(
            "swag",
            "regular",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    # endregion

    # region Load model config and tokenizer
    if checkpoint is not None:
        config_path = training_args.output_dir
    elif model_args.config_name:
        config_path = model_args.config_name
    else:
        config_path = model_args.model_name_or_path

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        config_path,
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
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_seq_length)
        # Un-flatten
        data = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return data

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if data_args.pad_to_max_length:
        data_collator = DefaultDataCollator(return_tensors="np")
    else:
        data_collator = DataCollatorForMultipleChoice(tokenizer, return_tensors="tf")
    # endregion

    with training_args.strategy.scope():
        # region Build model
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint
        model = TFAutoModelForMultipleChoice.from_pretrained(
            model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas

        if training_args.do_train:
            num_train_steps = (len(train_dataset) // total_train_batch_size) * int(training_args.num_train_epochs)
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
        else:
            optimizer = "sgd"  # Just write anything because we won't be using it
        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, metrics=["accuracy"], jit_compile=training_args.xla)
        # endregion

        # region Preparing push_to_hub and model card
        push_to_hub_model_id = training_args.push_to_hub_model_id
        model_name = model_args.model_name_or_path.split("/")[-1]
        if not push_to_hub_model_id:
            push_to_hub_model_id = f"{model_name}-finetuned-multiplechoice"

        model_card_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "multiple-choice"}

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

        # region Training
        eval_metrics = None
        if training_args.do_train:
            dataset_options = tf.data.Options()
            dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

            # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
            # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
            # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
            # yourself if you use this method, whereas they are automatically inferred from the model input names when
            # using model.prepare_tf_dataset()
            # For more info see the docs:
            # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
            # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset

            tf_train_dataset = model.prepare_tf_dataset(
                train_dataset,
                shuffle=True,
                batch_size=total_train_batch_size,
                collate_fn=data_collator,
            ).with_options(dataset_options)

            if training_args.do_eval:
                validation_data = model.prepare_tf_dataset(
                    eval_dataset,
                    shuffle=False,
                    batch_size=total_eval_batch_size,
                    collate_fn=data_collator,
                    drop_remainder=True,
                ).with_options(dataset_options)
            else:
                validation_data = None
            history = model.fit(
                tf_train_dataset,
                validation_data=validation_data,
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )
            eval_metrics = {key: val[-1] for key, val in history.history.items()}
        # endregion

        # region Evaluation
        if training_args.do_eval and not training_args.do_train:
            dataset_options = tf.data.Options()
            dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            # Do a standalone evaluation pass
            tf_eval_dataset = model.prepare_tf_dataset(
                eval_dataset,
                shuffle=False,
                batch_size=total_eval_batch_size,
                collate_fn=data_collator,
                drop_remainder=True,
            ).with_options(dataset_options)
            eval_results = model.evaluate(tf_eval_dataset)
            eval_metrics = {"val_loss": eval_results[0], "val_accuracy": eval_results[1]}
        # endregion

        if eval_metrics is not None and training_args.output_dir is not None:
            output_eval_file = os.path.join(training_args.output_dir, "all_results.json")
            with open(output_eval_file, "w") as writer:
                writer.write(json.dumps(eval_metrics))

        # region Push to hub

        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)
        # endregion


if __name__ == "__main__":
    main()
