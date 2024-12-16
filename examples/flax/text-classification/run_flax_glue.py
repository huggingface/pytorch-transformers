#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Finetuning a 🤗 Flax Transformers model for sequence classification on GLUE."""

import json
import logging
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import datasets
import evaluate
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import struct, traverse_util
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import HfApi
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    is_tensorboard_available,
)
from transformers.utils import check_min_version, send_example_telemetry


logger = logging.getLogger(__name__)
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any


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
    use_slow_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
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

    task_name: Optional[str] = field(
        default=None, metadata={"help": f"The name of the glue task to train on. choices {list(task_to_keys.keys())}"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
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
        if self.task_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower() if isinstance(self.task_name, str) else self.task_name


def create_train_state(
    model: FlaxAutoModelForSequenceClassification,
    learning_rate_fn: Callable[[int], float],
    is_regression: bool,
    num_labels: int,
    weight_decay: float,
) -> train_state.TrainState:
    """Create initial training state."""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          logits_fn: Applied to last layer to obtain the logits.
          loss_fn: Function to compute the loss.
        """

        logits_fn: Callable = struct.field(pytree_node=False)
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay, mask=decay_mask_fn
    )

    if is_regression:

        def mse_loss(logits, labels):
            return jnp.mean((logits[..., 0] - labels) ** 2)

        return TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=tx,
            logits_fn=lambda logits: logits[..., 0],
            loss_fn=mse_loss,
        )
    else:  # Classification.

        def cross_entropy_loss(logits, labels):
            xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
            return jnp.mean(xentropy)

        return TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=tx,
            logits_fn=lambda logits: logits.argmax(-1),
            loss_fn=cross_entropy_loss,
        )


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def glue_train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch


def glue_eval_data_collator(dataset: Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`. Sharding handled by `pad_shard_unpad` in the eval loop."""
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args, framework="flax")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if training_args.push_to_hub:
        # Retrieve of infer repo_name
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        # Create repo and retrieve repo_id
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            token=model_args.token,
        )
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (data_args.train_file if data_args.train_file is not None else data_args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=not model_args.use_slow_tokenizer,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=data_args.max_seq_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Define a summary writer
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(training_args.output_dir)
            summary_writer.hparams({**training_args.to_dict(), **vars(model_args), **vars(data_args)})
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    def write_train_metric(summary_writer, train_metrics, train_time, step):
        summary_writer.scalar("train_time", train_time, step)

        train_metrics = get_metrics(train_metrics)
        for key, vals in train_metrics.items():
            tag = f"train_{key}"
            for i, val in enumerate(vals):
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    def write_eval_metric(summary_writer, eval_metrics, step):
        for metric_name, value in eval_metrics.items():
            summary_writer.scalar(f"eval_{metric_name}", value, step)

    num_epochs = int(training_args.num_train_epochs)
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    train_batch_size = int(training_args.per_device_train_batch_size) * jax.local_device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    state = create_train_state(
        model, learning_rate_fn, is_regression, num_labels=num_labels, weight_decay=training_args.weight_decay
    )

    # define step functions
    def train_step(
        state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey
    ) -> Tuple[train_state.TrainState, float]:
        """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        targets = batch.pop("labels")

        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = state.loss_fn(logits, targets)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        metrics = jax.lax.pmean({"loss": loss, "learning_rate": learning_rate_fn(state.step)}, axis_name="batch")
        return new_state, metrics, new_dropout_rng

    p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

    def eval_step(state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        return state.logits_fn(logits)

    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    else:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    logger.info(f"===== Starting training ({num_epochs} epochs) =====")
    train_time = 0

    # make sure weights are replicated on each device
    state = replicate(state)

    steps_per_epoch = len(train_dataset) // train_batch_size
    total_steps = steps_per_epoch * num_epochs
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (0/{num_epochs})", position=0)
    for epoch in epochs:
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # train
        train_loader = glue_train_data_collator(input_rng, train_dataset, train_batch_size)
        for step, batch in enumerate(
            tqdm(
                train_loader,
                total=steps_per_epoch,
                desc="Training...",
                position=1,
            ),
        ):
            state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
            train_metrics.append(train_metric)

            cur_step = (epoch * steps_per_epoch) + (step + 1)

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step}/{total_steps} | Training Loss: {train_metric['loss']}, Learning Rate:"
                    f" {train_metric['learning_rate']})"
                )

                train_metrics = []

            if (cur_step % training_args.eval_steps == 0 or cur_step % steps_per_epoch == 0) and cur_step > 0:
                # evaluate
                eval_loader = glue_eval_data_collator(eval_dataset, eval_batch_size)
                for batch in tqdm(
                    eval_loader,
                    total=math.ceil(len(eval_dataset) / eval_batch_size),
                    desc="Evaluating ...",
                    position=2,
                ):
                    labels = batch.pop("labels")
                    predictions = pad_shard_unpad(p_eval_step)(
                        state, batch, min_device_batch=per_device_eval_batch_size
                    )
                    metric.add_batch(predictions=np.array(predictions), references=labels)

                eval_metric = metric.compute()

                logger.info(f"Step... ({cur_step}/{total_steps} | Eval metrics: {eval_metric})")

                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metric, cur_step)

            if (cur_step % training_args.save_steps == 0 and cur_step > 0) or (cur_step == total_steps):
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(unreplicate(state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        api.upload_folder(
                            commit_message=f"Saving weights and logs of epoch {epoch}",
                            folder_path=training_args.output_dir,
                            repo_id=repo_id,
                            repo_type="model",
                            token=training_args.hub_token,
                        )
            epochs.desc = f"Epoch ... {epoch + 1}/{num_epochs}"

    # save the eval metrics in json
    if jax.process_index() == 0:
        eval_metric = {f"eval_{metric_name}": value for metric_name, value in eval_metric.items()}
        path = os.path.join(training_args.output_dir, "eval_results.json")
        with open(path, "w") as f:
            json.dump(eval_metric, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
