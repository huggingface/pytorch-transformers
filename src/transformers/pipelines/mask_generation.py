from collections import defaultdict
from typing import Optional

from ..image_utils import load_image
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    logging,
    requires_backends,
)
from .base import ChunkPipeline, build_pipeline_init_args


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_MASK_GENERATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True),
    r"""
        points_per_batch (*optional*, int, default to 64):
            Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU
            memory.
        output_bboxes_mask (`bool`, *optional*, default to `False`):
            Whether or not to output the bounding box predictions.
        output_rle_masks (`bool`, *optional*, default to `False`):
            Whether or not to output the masks in `RLE` format""",
)
class MaskGenerationPipeline(ChunkPipeline):
    """
    Automatic mask generation for images using `SamForMaskGeneration`. This pipeline predicts binary masks for an
    image, given an image. It is a `ChunkPipeline` because you can seperate the points in a mini-batch in order to
    avoid OOM issues. Use the `points_per_batch` argument to control the number of points that will be processed at the
    same time. Default is `64`.

    The pipeline works in 3 steps:
        1. `preprocess`: A grid of 1024 points evenly separated is generated along with bounding boxes and point
           labels.
            For more details on how the points and bounding boxes are created, check the `_generate_crop_boxes`
            function. The image is also preprocessed using the `image_processor`. This function `yields` a minibatch of
            `points_per_batch`.

        2. `forward`: feeds the outputs of `preprocess` to the model. The image embedding is computed only once.
            Calls both `self.model.get_image_embeddings` and makes sure that the gradients are not computed, and the
            tensors and models are on the same device.

        3. `postprocess`: The most important part of the automatic mask generation happens here. Three steps
            are induced:
                - image_processor.postprocess_masks (run on each minibatch loop): takes in the raw output masks,
                  resizes them according
                to the image size, and transforms there to binary masks.
                - image_processor.filter_masks (on each minibatch loop): uses both `pred_iou_thresh` and
                  `stability_scores`. Also
                applies a variety of filters based on non maximum suppression to remove bad masks.
                - image_processor.postprocess_masks_for_amg applies the NSM on the mask to only keep relevant ones.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")
    >>> outputs = generator(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ... )

    >>> outputs = generator(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", points_per_batch=128
    ... )
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"mask-generation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=mask-generation).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        requires_backends(self, "vision")
        requires_backends(self, "torch")

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        self.check_model_type(MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        forward_params = {}
        # preprocess args
        if "points_per_batch" in kwargs:
            preprocess_kwargs["points_per_batch"] = kwargs["points_per_batch"]
        if "points_per_crop" in kwargs:
            preprocess_kwargs["points_per_crop"] = kwargs["points_per_crop"]
        if "crops_n_layers" in kwargs:
            preprocess_kwargs["crops_n_layers"] = kwargs["crops_n_layers"]
        if "crop_overlap_ratio" in kwargs:
            preprocess_kwargs["crop_overlap_ratio"] = kwargs["crop_overlap_ratio"]
        if "crop_n_points_downscale_factor" in kwargs:
            preprocess_kwargs["crop_n_points_downscale_factor"] = kwargs["crop_n_points_downscale_factor"]
        if "timeout" in kwargs:
            preprocess_kwargs["timeout"] = kwargs["timeout"]
        # postprocess args
        if "pred_iou_thresh" in kwargs:
            forward_params["pred_iou_thresh"] = kwargs["pred_iou_thresh"]
        if "stability_score_offset" in kwargs:
            forward_params["stability_score_offset"] = kwargs["stability_score_offset"]
        if "mask_threshold" in kwargs:
            forward_params["mask_threshold"] = kwargs["mask_threshold"]
        if "stability_score_thresh" in kwargs:
            forward_params["stability_score_thresh"] = kwargs["stability_score_thresh"]
        if "crops_nms_thresh" in kwargs:
            postprocess_kwargs["crops_nms_thresh"] = kwargs["crops_nms_thresh"]
        if "output_rle_mask" in kwargs:
            postprocess_kwargs["output_rle_mask"] = kwargs["output_rle_mask"]
        if "output_bboxes_mask" in kwargs:
            postprocess_kwargs["output_bboxes_mask"] = kwargs["output_bboxes_mask"]
        return preprocess_kwargs, forward_params, postprocess_kwargs

    def __call__(self, image, *args, num_workers=None, batch_size=None, **kwargs):
        """
        Generates binary segmentation masks

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                Image or list of images.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold to use when turning the predicted masks into binary values.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                A filtering threshold in `[0,1]` applied on the model's predicted mask quality.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                A filtering threshold in `[0,1]`, using the stability of the mask under changes to the cutoff used to
                binarize the model's mask predictions.
            stability_score_offset (`int`, *optional*, defaults to 1):
                The amount to shift the cutoff when calculated the stability score.
            crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
            crops_n_layers (`int`, *optional*, defaults to 0):
                If `crops_n_layers>0`, mask prediction will be run again on crops of the image. Sets the number of
                layers to run, where each layer has 2**i_layer number of image crops.
            crop_overlap_ratio (`float`, *optional*, defaults to `512 / 1500`):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            crop_n_points_downscale_factor (`int`, *optional*, defaults to `1`):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            `Dict`: A dictionary with the following keys:
                - **mask** (`PIL.Image`) -- A binary mask of the detected object as a PIL Image of shape `(width,
                  height)` of the original image. Returns a mask filled with zeros if no object is found.
                - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of
                  the "object" described by the label and the mask.

        """
        return super().__call__(image, *args, num_workers=num_workers, batch_size=batch_size, **kwargs)

    def preprocess(
        self,
        image,
        points_per_batch=64,
        crops_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[int] = 1,
        timeout: Optional[float] = None,
    ):
        image = load_image(image, timeout=timeout)
        target_size = self.image_processor.size["longest_edge"]
        crop_boxes, grid_points, cropped_images, input_labels = self.image_processor.generate_crop_boxes(
            image, target_size, crops_n_layers, crop_overlap_ratio, points_per_crop, crop_n_points_downscale_factor
        )
        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt")
        if self.framework == "pt":
            model_inputs = model_inputs.to(self.torch_dtype)

        with self.device_placement():
            if self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))
                    model_inputs["image_embeddings"] = image_embeddings

        # When using crops, each crop has its own set of points
        if crops_n_layers > 0:
            n_crops = len(cropped_images)
            for crop_idx in range(n_crops):
                crop_points = grid_points[crop_idx:crop_idx+1]  # Keep batch dimension
                crop_labels = input_labels[crop_idx:crop_idx+1]  # Keep batch dimension
                n_points = crop_points.shape[2]  # Number of points for this crop
                points_per_batch = points_per_batch if points_per_batch is not None else n_points

                if points_per_batch <= 0:
                    raise ValueError(
                        "Cannot have points_per_batch<=0. Must be >=1 to returned batched outputs. "
                        "To return all points at once, set points_per_batch to None"
                    )

                for i in range(0, n_points, points_per_batch):
                    batched_points = crop_points[:, :, i:i + points_per_batch, :]
                    labels = crop_labels[:, :, i:i + points_per_batch]
                    is_last = (crop_idx == n_crops - 1) and (i == n_points - points_per_batch)
                    yield {
                        "input_points": batched_points,
                        "input_labels": labels,
                        "input_boxes": crop_boxes[crop_idx:crop_idx+1],
                        "is_last": is_last,
                        "image_embeddings": image_embeddings[crop_idx:crop_idx+1],
                        **{k: v[crop_idx:crop_idx+1] if isinstance(v, torch.Tensor) else v
                           for k, v in model_inputs.items() if k != "image_embeddings"}
                    }
        else:
            # Original behavior for no crops
            n_points = grid_points.shape[2]
            points_per_batch = points_per_batch if points_per_batch is not None else n_points

            if points_per_batch <= 0:
                raise ValueError(
                    "Cannot have points_per_batch<=0. Must be >=1 to returned batched outputs. "
                    "To return all points at once, set points_per_batch to None"
                )

            for i in range(0, n_points, points_per_batch):
                batched_points = grid_points[:, :, i:i + points_per_batch, :]
                labels = input_labels[:, :, i:i + points_per_batch]
                is_last = i == n_points - points_per_batch
                yield {
                    "input_points": batched_points,
                    "input_labels": labels,
                    "input_boxes": crop_boxes,
                    "is_last": is_last,
                    **model_inputs,
                }

    def _forward(
        self,
        model_inputs,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        input_boxes = model_inputs.pop("input_boxes")
        is_last = model_inputs.pop("is_last")
        original_sizes = model_inputs.pop("original_sizes").tolist()
        reshaped_input_sizes = model_inputs.pop("reshaped_input_sizes").tolist()

        model_outputs = self.model(**model_inputs)

        # post processing happens here in order to avoid CPU GPU copies of ALL the masks
        low_resolution_masks = model_outputs["pred_masks"]
        masks = self.image_processor.post_process_masks(
            low_resolution_masks, original_sizes, reshaped_input_sizes, mask_threshold, binarize=False
        )
        iou_scores = model_outputs["iou_scores"]
        masks, iou_scores, boxes = self.image_processor.filter_masks(
            masks[0],
            iou_scores[0],
            original_sizes[0],
            input_boxes[0],
            pred_iou_thresh,
            stability_score_thresh,
            mask_threshold,
            stability_score_offset,
        )
        return {
            "masks": masks,
            "is_last": is_last,
            "boxes": boxes,
            "iou_scores": iou_scores,
        }

    def postprocess(
        self,
        model_outputs,
        output_rle_mask=False,
        output_bboxes_mask=False,
        crops_nms_thresh=0.7,
    ):
        all_scores = []
        all_masks = []
        all_boxes = []
        for model_output in model_outputs:
            all_scores.append(model_output.pop("iou_scores"))
            all_masks.extend(model_output.pop("masks"))
            all_boxes.append(model_output.pop("boxes"))

        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        output_masks, iou_scores, rle_mask, bounding_boxes = self.image_processor.post_process_for_mask_generation(
            all_masks, all_scores, all_boxes, crops_nms_thresh
        )

        extra = defaultdict(list)
        for output in model_outputs:
            for k, v in output.items():
                extra[k].append(v)

        optional = {}
        if output_rle_mask:
            optional["rle_mask"] = rle_mask

        if output_bboxes_mask:
            optional["bounding_boxes"] = bounding_boxes

        return {"masks": output_masks, "scores": iou_scores, **optional, **extra}


