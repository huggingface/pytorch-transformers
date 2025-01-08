from typing import List, Tuple, Union

from transformers.models.detr.image_processing_detr_fast import DetrImageProcessorFast

from ...image_transforms import center_to_corners_format
from ...utils import (
    TensorType,
    is_torch_available,
    logging,
)


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class DeformableDetrImageProcessorFast(DetrImageProcessorFast):
    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DeformableDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None, top_k: int = 100
    ):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            top_k (`int`, *optional*, defaults to 100):
                Keep only top k bounding boxes before filtering by thresholding.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = out_logits.sigmoid()
        prob = prob.view(out_logits.shape[0], -1)
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    def post_process_segmentation():
        raise NotImplementedError("Segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_instance():
        raise NotImplementedError("Instance post-processing is not implemented for Deformable DETR yet.")

    def post_process_panoptic():
        raise NotImplementedError("Panoptic post-processing is not implemented for Deformable DETR yet.")

    def post_process_instance_segmentation():
        raise NotImplementedError("Segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_semantic_segmentation():
        raise NotImplementedError("Semantic segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_panoptic_segmentation():
        raise NotImplementedError("Panoptic segmentation post-processing is not implemented for Deformable DETR yet.")


__all__ = ["DeformableDetrImageProcessorFast"]
