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

import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .. import RelationDetrConfig
from ..image_transforms import center_to_corners_format
from ..utils import is_scipy_available, requires_backends
from .loss_for_object_detection import (
    _set_aux_loss,
    box_iou,
    generalized_box_iou,
)


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


def reduce_loss(loss, reduction: str = "sum"):
    if reduction == "none":
        return loss

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        num_elements = max(loss.numel(), 1)
        return loss.sum() / num_elements

    raise ValueError("Only sum, mean and none are valid reduction")


# Modified from from transformers.loss.loss_for_object_detection.py, remove num_boxes and rewrite code logic
def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2, reduction: str = "sum"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        reduction (`str`, *optional*, defaults to "sum"):
            Specifies the reduction to apply to the output

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    target_score = targets.to(inputs.dtype)
    pos_weight = alpha * (1 - prob).pow(gamma)
    neg_weight = (1 - alpha) * prob.pow(gamma)
    weight = neg_weight * (1 - targets) + targets * pos_weight
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    loss = loss * weight  # keep gradient on weight according to Deformable-DETR
    return reduce_loss(loss, reduction)


def score_aware_vari_sigmoid_focal_loss(
    inputs, targets, gt_score, alpha: float = 0.25, gamma: float = 2, reduction: str = "sum"
):
    """
    Computes the VariFocalLoss with binary style and gt_score as positive modulation.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the inputs.
        gt_score (`torch.FloatTensor` with the same shape as `inputs`):
            A tensor storing the score for each element in the inputs (typically IoU), used as weight to
            modulate positive supervision.
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`float`, *optional*, defaults to 2):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        reduction (`str`, *optional*, defaults to "sum"):
            Specifies the reduction to apply to the output.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid().detach()
    target_score = targets * gt_score
    # NOTE: The only difference from sigmoid_focal_loss
    pos_weight = target_score  # pos_weight = gt_score.unsqueeze(-1)
    neg_weight = (1 - alpha) * prob.pow(gamma)
    weight = neg_weight * (1 - targets) + targets * pos_weight
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    return reduce_loss(loss, reduction)


class RelationDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost (`float`, default=1.0):
            The relative weight of the classification error in the matching cost.
        bbox_cost (`float`, default=1.0):
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost (`float`, default=1.0):
            The relative weight of the giou loss of the bounding box in the matching cost.
        focal_alpha (`float`, default=0.25):
            Alpha in Focal Loss.
        focal_gamma (`float`, default=2.0):
            Gamma in Focal Loss.
    """

    def __init__(
        self,
        class_cost: float = 1.0,
        bbox_cost: float = 1.0,
        giou_cost: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def calculate_cost(
        self,
        pred_boxes: Tensor,
        pred_logits: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ):
        # Compute the classification cost.
        pred_logits = pred_logits[:, gt_labels]
        neg_cost = sigmoid_focal_loss(
            pred_logits,
            torch.zeros_like(pred_logits),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )
        pos_cost = sigmoid_focal_loss(
            pred_logits,
            torch.ones_like(pred_logits),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )
        class_cost = (pos_cost - neg_cost) * self.class_cost

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(pred_boxes, gt_boxes, p=1) * self.bbox_cost

        # Compute the giou cost betwen boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(pred_boxes), center_to_corners_format(gt_boxes))
        giou_cost = giou_cost * self.giou_cost

        cost = class_cost + bbox_cost + giou_cost

        # Final cost matrix
        return cost

    @torch.no_grad()
    def forward(self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor):
        matching_cost = self.calculate_cost(
            pred_boxes=pred_boxes, pred_logits=pred_logits, gt_boxes=gt_boxes, gt_labels=gt_labels
        )
        indices = linear_sum_assignment(matching_cost.cpu())
        src_ind = torch.as_tensor(indices[0], device=pred_logits.device)
        tgt_ind = torch.as_tensor(indices[1], device=pred_logits.device)
        return src_ind, tgt_ind


class RelationDetrLoss(nn.Module):
    """
    This class computes the loss for RelationDetrForObjectDetection. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        matcher (`RelationDetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        loss_class (`float`):
            Loss weight for the class classification loss.
        loss_bbox (`float`):
            Loss weight for box localization loss.
        loss_giou (`float`):
            Loss weight for generalized IoU loss.
        alpha (`float`):
            Alpha in Focal Loss.
        gamma (`float`):
            Gamma in Focal Loss.
        two_stage_binary_cls (`bool`):
            Whether to use two-stage binary classification loss.
    """

    def __init__(self, config: RelationDetrConfig):
        super().__init__()
        self.num_classes = config.num_labels
        self.matcher = RelationDetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
        )
        self.loss_class = config.class_loss_coefficient
        self.loss_bbox = config.bbox_loss_coefficient
        self.loss_giou = config.giou_loss_coefficient
        self.alpha = config.focal_alpha
        self.gamma = config.focal_gamma
        self.two_stage_binary_cls = config.two_stage_binary_cls

    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        """
        Computing the loss related to labels, VariFocalLoss. Targets dicts must contain the key "class_labels"
        containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            box_iou(
                center_to_corners_format(src_boxes),
                center_to_corners_format(target_boxes),
            )[0]
        ).detach()  # add detach according to RT-DETR

        assert "logits" in outputs
        src_logits = outputs["logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        loss_class = (
            score_aware_vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score.unsqueeze(-1),
                alpha=self.alpha,
                gamma=self.gamma,
            )
            / num_boxes
        )
        losses = {"loss_class": loss_class * self.loss_class}
        return losses

    def loss_boxes(self, outputs, targets, num_boxes, indices, **kwargs):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes * self.loss_bbox

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(src_boxes),
                center_to_corners_format(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes * self.loss_giou
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_loss(self, outputs, targets, num_boxes, indices=None, **kwargs):
        losses = {}
        # get matching results for each image
        if not indices:
            gt_boxes = [t["boxes"] for t in targets]
            gt_labels = [t["class_labels"] for t in targets]
            logits, pred_boxes = outputs["logits"], outputs["pred_boxes"]
            indices = list(map(self.matcher, pred_boxes, logits, gt_boxes, gt_labels))

        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
        losses.update(loss_class)
        losses.update(loss_boxes)
        return losses

    def auxiliary_loss(self, aux_outputs, *args, **kwargs):
        losses = {}
        for i, aux_output in enumerate(aux_outputs):
            # get matching results for each image
            losses_aux = self.calculate_loss(aux_output, *args, **kwargs)
            losses.update({k + f"_{i}": v for k, v in losses_aux.items()})
        return losses

    def enc_loss(self, enc_outputs, targets, *args, **kwargs):
        bin_targets = copy.deepcopy(targets)
        if self.two_stage_binary_cls:
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
        losses_enc = self.calculate_loss(enc_outputs, bin_targets, *args, **kwargs)
        return {k + "_enc": v for k, v in losses_enc.items()}

    def forward(self, outputs, targets, indices=None):
        """
        This performs the loss computation.

        Args:
             outputs: (`dict`):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        if "logits" in outputs and "pred_boxes" in outputs:
            losses.update(self.calculate_loss(outputs, targets, num_boxes, indices=indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            losses.update(self.auxiliary_loss(outputs["auxiliary_outputs"], targets, num_boxes, indices=indices))

        if "encoder_outputs" in outputs:
            losses.update(self.enc_loss(outputs["encoder_outputs"], targets, num_boxes, indices=indices))

        return losses


def RelationDetrForObjectDetectionLoss(
    labels,
    config,
    outputs_class=None,
    outputs_coord=None,
    enc_topk_logits=None,
    enc_topk_bboxes=None,
    denoising_meta_values=None,
    **kwargs,
):
    criterion = RelationDetrLoss(config)
    device = outputs_class.device
    criterion.to(device)

    if denoising_meta_values is not None:
        dn_out_coord, outputs_coord = torch.split(outputs_coord, denoising_meta_values["dn_num_split"], dim=2)
        dn_out_class, outputs_class = torch.split(outputs_class, denoising_meta_values["dn_num_split"], dim=2)

    # Second: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = outputs_class[:, -1]
    outputs_loss["pred_boxes"] = outputs_coord[:, -1]
    auxiliary_outputs = _set_aux_loss(outputs_class.transpose(0, 1), outputs_coord.transpose(0, 1))
    outputs_loss["auxiliary_outputs"] = auxiliary_outputs
    outputs_loss["encoder_outputs"] = {"logits": enc_topk_logits, "pred_boxes": enc_topk_bboxes}

    loss_dict = criterion(outputs_loss, labels)

    if denoising_meta_values is not None:
        dn_outputs_loss = {}
        dn_outputs_loss["logits"] = dn_out_class[:, -1]
        dn_outputs_loss["pred_boxes"] = dn_out_coord[:, -1]
        dn_outputs_loss["auxiliary_outputs"] = _set_aux_loss(
            dn_out_class.transpose(0, 1), dn_out_coord.transpose(0, 1)
        )

        dn_idx = []
        dn_num_group = denoising_meta_values["dn_num_group"]
        max_gt_num_per_image = denoising_meta_values["max_gt_num_per_image"]
        for i in range(len(labels)):
            if len(labels[i]["class_labels"]) > 0:
                group_index, target_index = torch.meshgrid(
                    torch.arange(dn_num_group, device=device),
                    torch.arange(len(labels[i]["class_labels"]), device=device),
                    indexing="ij",
                )
                output_idx = group_index * max_gt_num_per_image + target_index
                output_idx = output_idx.flatten()
                tgt_idx = target_index.flatten()
            else:
                output_idx = tgt_idx = torch.tensor([], dtype=torch.long, device=device)
            dn_idx.append((output_idx, tgt_idx))

        dn_loss_dict = criterion(dn_outputs_loss, labels, indices=dn_idx)
        loss_dict.update({k + "_dn": v / dn_num_group for k, v in dn_loss_dict.items()})

    loss = sum(loss_dict.values())
    if isinstance(auxiliary_outputs, list):
        auxiliary_outputs = tuple(auxiliary_outputs)
    return loss, loss_dict, auxiliary_outputs
