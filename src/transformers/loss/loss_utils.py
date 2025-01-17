# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss

from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
from .loss_for_object_detection import ForObjectDetectionLoss, ForSegmentationLoss
from .loss_rt_detr import RTDetrForObjectDetectionLoss


def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def ForMaskedLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    # Enable model parallelism

    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def ForSequenceClassificationLoss(labels, pooled_logits, config, **kwargs):
    num_labels = config.num_labels
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    labels = labels.to(pooled_logits.device)
    if config.problem_type == "regression":
        loss_fct = MSELoss()
        if num_labels == 1:
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            loss = loss_fct(pooled_logits, labels)
    elif config.problem_type == "single_label_classification":
        loss = fixed_cross_entropy(pooled_logits.view(-1, num_labels), labels.view(-1), **kwargs)
    elif config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)
    return loss


def ForQuestionAnsweringLoss(start_logits, end_logits, start_positions, end_positions, **kwargs):
    total_loss = None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1).to(start_logits.device)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1).to(end_logits.device)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        start_loss = fixed_cross_entropy(start_logits, start_positions, ignore_index=ignored_index, **kwargs)
        end_loss = fixed_cross_entropy(end_logits, end_positions, ignore_index=ignored_index, **kwargs)
        total_loss = (start_loss + end_loss) / 2
    return total_loss


def ForTokenClassification(logits, labels, config, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.view(-1, config.num_labels)
    labels = labels.view(-1).to(logits.device)
    logits = logits.float()
    # Flatten the tokens
    return fixed_cross_entropy(logits, labels, **kwargs)


LOSS_MAPPING = {
    "ForCausalLM": ForCausalLMLoss,
    "ForMaskedLM": ForMaskedLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
    "ForSegmentation": ForSegmentationLoss,
    "ForObjectDetection": ForObjectDetectionLoss,
    "DeformableDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "ConditionalDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "GroundingDinoForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "ConditionalDetrForSegmentation": DeformableDetrForSegmentationLoss,
    "RTDetrForObjectDetection": RTDetrForObjectDetectionLoss,
}
