#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Finetuning 🤗 Transformers model for instance segmentation leveraging the Trainer API."""

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/instance-segmentation/requirements.txt")


@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    model_name_or_path: str = field(
        default="facebook/mask2former-swin-tiny-coco-instance",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    dataset_name: str = field(
        default="qubvel-hf/ade20k-mini",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
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
    image_height: Optional[int] = field(default=512, metadata={"help": "Image height after resizing."})
    image_width: Optional[int] = field(default=512, metadata={"help": "Image width after resizing."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    do_reduce_labels: bool = field(
        default=False,
        metadata={
            "help": (
                "If background class is labeled as 0 and you want to remove it from the labels, set this flag to True."
            )
        },
    )


def augment_and_transform_batch(
    examples: Mapping[str, Any], transform: A.Compose, image_processor: AutoImageProcessor
) -> BatchFeature:
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }

    for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
        image = np.array(pil_image)
        semantic_and_instance_masks = np.array(pil_annotation)[..., :2]

        # Apply augmentations
        output = transform(image=image, mask=semantic_and_instance_masks)

        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

        # Create mapping from instance id to semantic id
        unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {
            instance_id: semantic_id for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
        }

        # Apply the image processor transformations: resizing, rescaling, normalization
        model_inputs = image_processor(
            images=[aug_image],
            segmentation_maps=[aug_instance_mask],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )

        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])

    return batch


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch


@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


class Evaluator:
    """
    Compute metrics for the instance segmentation task.
    """

    def __init__(
        self,
        image_processor: AutoImageProcessor,
        id2label: Mapping[int, str],
        threshold: float = 0.0,
    ):
        """
        Initialize evaluator with image processor, id2label mapping and threshold for filtering predictions.

        Args:
            image_processor (AutoImageProcessor): Image processor for
                `post_process_instance_segmentation` method.
            id2label (Mapping[int, str]): Mapping from class id to class name.
            threshold (float): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        """
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = self.get_metric()

    def get_metric(self):
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        return metric

    def reset_metric(self):
        self.metric.reset()

    def postprocess_target_batch(self, target_batch) -> List[Dict[str, torch.Tensor]]:
        """Collect targets in a form of list of dictionaries with keys "masks", "labels"."""
        batch_masks = target_batch[0]
        batch_labels = target_batch[1]
        post_processed_targets = []
        for masks, labels in zip(batch_masks, batch_labels):
            post_processed_targets.append(
                {
                    "masks": masks.to(dtype=torch.bool),
                    "labels": labels,
                }
            )
        return post_processed_targets

    def get_target_sizes(self, post_processed_targets) -> List[List[int]]:
        target_sizes = []
        for target in post_processed_targets:
            target_sizes.append(target["masks"].shape[-2:])
        return target_sizes

    def postprocess_prediction_batch(self, prediction_batch, target_sizes) -> List[Dict[str, torch.Tensor]]:
        """Collect predictions in a form of list of dictionaries with keys "masks", "labels", "scores"."""

        model_output = ModelOutput(class_queries_logits=prediction_batch[0], masks_queries_logits=prediction_batch[1])
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        post_processed_predictions = []
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            if image_predictions["segments_info"]:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]),
                }
            else:
                # for void predictions, we need to provide empty tensors
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)

        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        """
        Update metrics with current evaluation results and return metrics if `compute_result` is True.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            compute_result (bool): Whether to compute and return metrics.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        # For metric computation we need to provide:
        #  - targets in a form of list of dictionaries with keys "masks", "labels"
        #  - predictions in a form of list of dictionaries with keys "masks", "labels", "scores"
        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        # Compute metrics
        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        # Reset metric for next evaluation
        self.reset_metric()

        return metrics


def setup_logging(training_args: TrainingArguments) -> None:
    """Setup logging according to `training_args`."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def find_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    """Find the last checkpoint in the output directory according to parameters specified in `training_args`."""

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
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

    return checkpoint


def main():
    # See all possible arguments in https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    # or by passing the --help flag to this script.

    parser = HfArgumentParser([Arguments, TrainingArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    # Set default training arguments for instance segmentation
    training_args.eval_do_concat_batches = False
    training_args.batch_eval_metrics = True
    training_args.remove_unused_columns = False

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_instance_segmentation", args)

    # Setup logging and log on each process the small summary:
    setup_logging(training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load last checkpoint from output_dir if it exists (and we are not overwriting it)
    checkpoint = find_last_checkpoint(training_args)

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(args.dataset_name, trust_remote_code=args.trust_remote_code)

    # We need to specify the label2id mapping for the model
    # it is a mapping from semantic class name to class index.
    # In case your dataset does not provide it, you can create it manually:
    # label2id = {"background": 0, "cat": 1, "dog": 2}
    label2id = dataset["train"][0]["semantic_class_to_id"]

    if args.do_reduce_labels:
        label2id = {name: idx for name, idx in label2id.items() if idx != 0}  # remove background class
        label2id = {name: idx - 1 for name, idx in label2id.items()}  # shift class indices by -1

    id2label = {v: k for k, v in label2id.items()}

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------
    model = AutoModelForUniversalSegmentation.from_pretrained(
        args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=args.token,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        do_resize=True,
        size={"height": args.image_height, "width": args.image_width},
        do_reduce_labels=args.do_reduce_labels,
        reduce_labels=args.do_reduce_labels,  # TODO: remove when mask2former support `do_reduce_labels`
        token=args.token,
    )

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
    train_augment_and_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
    )
    validation_transform = A.Compose(
        [A.NoOp()],
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    compute_metrics = Evaluator(image_processor=image_processor, id2label=id2label, threshold=0.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": args.model_name_or_path,
        "dataset": args.dataset_name,
        "tags": ["image-segmentation", "instance-segmentation", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
