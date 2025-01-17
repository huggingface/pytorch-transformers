import torch
import torch.nn as nn

from ..image_transforms import center_to_corners_format
from ..utils import is_scipy_available
from .loss_for_object_detection import HungarianMatcher, ImageLoss, _set_aux_loss, generalized_box_iou


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


# Similar to the one used in `DeformableDetr` but we reduce with sum and normalize by num_boxes
# instead of mean. It's worth noting that `num_queries` can't be obtained from `inputs`
# as `logits` are flattened due to masked selection.
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`int`):
            The total number of boxes in the batch.
        num_queries (`int`):
            The number of query boxes per image.
        alpha (`float`, *optional*, defaults to 0.25):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to 2):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


class GroundingDinoHungarianMatcher(HungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
                * "label_maps": Tuple of tensors of dim [num_classes, hidden_dim].
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, hidden_dim]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        label_maps = outputs["label_maps"]

        # First take the label map for each class in each batch and then concatenate them
        label_maps = torch.cat([label_map[target["class_labels"]] for label_map, target in zip(label_maps, targets)])
        # Normalize label maps based on number of tokens per class
        label_maps = label_maps / label_maps.sum(dim=-1, keepdim=True)

        # Also concat the target labels and boxes
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # Compute the classification cost by taking pos and neg cost in the appropriate index
        class_cost = (pos_cost_class - neg_cost_class) @ label_maps.t()

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class GroundingDinoImageLoss(ImageLoss):
    """
    This class computes the losses for `GroundingDinoForObjectDetection`. The process happens in two steps: 1) we
    compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
    matched ground-truth / prediction (supervise class and box).

    Args:
        matcher (`GroundingDinoHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, focal_alpha, losses):
        nn.Module.__init__(self)
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.losses = losses

    def _get_target_classes_one_hot(self, outputs, targets, indices):
        """
        Create one_hot based on the matching indices
        """
        logits = outputs["logits"]
        # Add offsets to class_labels to select the correct label map
        class_labels = torch.cat(
            [
                target["class_labels"][J] + len(outputs["label_maps"][i]) if i > 0 else target["class_labels"][J]
                for i, (target, (_, J)) in enumerate(zip(targets, indices))
            ]
        )
        label_maps = torch.cat(outputs["label_maps"], dim=0)

        idx = self._get_source_permutation_idx(indices)
        target_classes_onehot = torch.zeros_like(logits, device=logits.device, dtype=torch.long)
        target_classes_onehot[idx] = label_maps[class_labels].to(torch.long)

        return target_classes_onehot

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        if "text_mask" not in outputs:
            raise KeyError("No text_mask were found in the outputs")

        target_classes_onehot = self._get_target_classes_one_hot(outputs, targets, indices)
        source_logits = outputs["logits"]
        text_mask = outputs["text_mask"]

        # Select only valid logits
        source_logits = torch.masked_select(source_logits, text_mask)
        target_classes_onehot = torch.masked_select(target_classes_onehot, text_mask)

        num_queries = source_logits.shape[0]

        target_classes_onehot = target_classes_onehot.float()
        loss_ce = sigmoid_focal_loss(
            inputs=source_logits,
            targets=target_classes_onehot,
            num_boxes=num_boxes,
            num_queries=num_queries,
            alpha=self.focal_alpha,
            gamma=2,
        )

        losses = {"loss_ce": loss_ce}

        return losses


def GroundingDinoForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    label_maps,
    text_mask,
    outputs_class=None,
    outputs_coord=None,
    encoder_logits=None,
    encoder_pred_boxes=None,
):
    # First: create the matcher
    matcher = GroundingDinoHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = GroundingDinoImageLoss(
        matcher=matcher,
        class_reduction=config.class_loss_reduction,
        focal_alpha=config.focal_alpha,
        losses=losses,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["label_maps"] = label_maps
    outputs_loss["text_mask"] = text_mask

    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        for aux_output in auxiliary_outputs:
            aux_output["label_maps"] = label_maps
            aux_output["text_mask"] = text_mask
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    if config.two_stage:
        outputs_loss["enc_outputs"] = {
            "logits": encoder_logits,
            "pred_boxes": encoder_pred_boxes,
            "label_maps": label_maps,
            "text_mask": text_mask,
        }

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": config.bbox_loss_coefficient,
        "loss_giou": config.giou_loss_coefficient,
    }

    if config.two_stage:
        enc_weight_dict = {k + "_enc": v for k, v in weight_dict.items()}
        weight_dict.update(enc_weight_dict)

    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
