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
# limitations under the License.
"""Image processor class for Sapiens."""

from typing import Dict, List, Optional, Union, Tuple

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    is_torch_tensor,
    is_torch_available,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def interpolate_if_needed(logits, target_sizes, mode: str, align_corners: bool) -> List["torch.Tensor"]:
    """
    Interpolates logits if target sizes are provided and logit size is different from target size.
    """
    
    if target_sizes is None:
        return [logits_i for logits_i in logits]

    if len(logits) != len(target_sizes):
        raise ValueError(
            "Make sure that you pass in as many target sizes as the batch dimension of the logits"
        )

    if is_torch_tensor(target_sizes):
        target_sizes = target_sizes.numpy()

    resized_logits = []
    
    for logits_i, target_size in zip(logits, target_sizes):    
        
        src_height, src_width = logits_i.shape[1:]
        dst_height, dst_width = target_size

        if src_height == dst_height and src_width == dst_width:
            resized_logits_i = logits_i
        else:
            resized_logits_i = torch.nn.functional.interpolate(
                logits_i.unsqueeze(dim=0), size=target_size, mode=mode, align_corners=align_corners
            ).squeeze(dim=0)
        
        resized_logits.append(resized_logits_i)

    return resized_logits


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """

    assert isinstance(heatmaps, np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        num_keypoints, height, width = heatmaps.shape
        batch_size = None
        heatmaps_flatten = heatmaps.reshape(num_keypoints, -1)
    else:
        batch_size, num_keypoints, height, width = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(batch_size * num_keypoints, -1)

    indexes = np.argmax(heatmaps_flatten, axis=1)
    y_coordinate, x_coordinate = np.unravel_index(indexes, shape=(height, width))
    
    keypoints = np.stack((x_coordinate, y_coordinate), axis=-1).astype(np.float32)
    scores = np.amax(heatmaps_flatten, axis=1)
    keypoints[scores <= 0] = -1

    if batch_size is not None:
        keypoints = keypoints.reshape(batch_size, num_keypoints, 2)
        scores = scores.reshape(batch_size, num_keypoints)

    return keypoints, scores


class SapiensImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Sapiens image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        data = {"pixel_values": []}

        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size_dict, resample=resample, input_data_format=input_data_format)
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

            data["pixel_values"].append(image)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_semantic_segmentation(self, outputs, target_sizes: Optional[List[Tuple]] = None):
        """
        Converts the output of [`SapiensForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`SapiensForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        logits = outputs.logits
        logits = interpolate_if_needed(logits, target_sizes, mode="bilinear", align_corners=False)
        segmentation_maps = [logits_i.argmax(dim=0) for logits_i in logits]
        return segmentation_maps

    def post_process_normal_estimation(
            self, 
            outputs,
            target_sizes: Optional[List[Tuple]] = None,
            segmentation_maps: Optional[List[torch.Tensor]] = None,
        ):
        """
        Converts the output of [`SapiensForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`SapiensForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            segmentation_maps (`List[torch.Tensor]` of length `batch_size`, *optional*):


        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        
        logits = outputs.logits
        logits = interpolate_if_needed(logits, target_sizes, mode="bilinear", align_corners=False)

        if segmentation_maps is not None:
            if len(segmentation_maps) != len(logits):
                raise ValueError(
                    f"Make sure that you pass in as many segmentation maps as the batch dimension of the logits: {len(logits)}"
                )
            segmentation_maps = interpolate_if_needed(segmentation_maps, target_sizes, mode="nearest", align_corners=False)

        output = []
        for i, logits_i in enumerate(logits):
            
            norm = logits_i.norm(dim=0, keepdim=True)
            normal_map = logits_i / (norm + 1e-5)
            normal_map = normal_map.permute(1, 2, 0)

            rgb_normal_map = (normal_map + 1.0) * 127.5
            rgb_normal_map = rgb_normal_map[..., ::-1].to(torch.uint8)

            if segmentation_maps is not None:
                mask = segmentation_maps[i].unsqueeze(-1) == 0
                mask = mask.to(rgb_normal_map.device)
                rgb_normal_map[mask] = 0
            
            normal_map = normal_map.cpu().numpy()
            rgb_normal_map = rgb_normal_map.cpu().numpy()

            output.append({
                "normal_map": normal_map,
                "normal_map_rgb": rgb_normal_map,
            })

        return output
    
    def post_process_depth_estimation(
            self, 
            outputs,
            target_sizes: Optional[List[Tuple]] = None,
            segmentation_maps: Optional[List[torch.Tensor]] = None,
        ):
        """
        Converts the output of [`SapiensForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`SapiensForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            segmentation_maps (`List[torch.Tensor]` of length `batch_size`, *optional*):


        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        
        logits = outputs.logits
        logits = interpolate_if_needed(logits, target_sizes, mode="bilinear", align_corners=False)

        if segmentation_maps is not None:
            if len(segmentation_maps) != len(logits):
                raise ValueError(
                    f"Make sure that you pass in as many segmentation maps as the batch dimension of the logits: {len(logits)}"
                )
            segmentation_maps = interpolate_if_needed(segmentation_maps, target_sizes, mode="nearest", align_corners=False)

        output = []
        for i, logits_i in enumerate(logits):
            
            depth_map = logits_i.squeeze(0)
            depth_map = depth_map.cpu().numpy()

            if segmentation_maps is not None:
                mask = segmentation_maps[i] == 0
                mask = mask.cpu().numpy()

                depth_foreground = depth_map[mask]
                normalized_depth = np.full_like(mask, 0, dtype=np.uint8)

                # normalize by foreground to range 0..1 and invert
                if len(depth_foreground) > 0:
                    min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
                    depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val))
                    depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)
                    normalized_depth[mask] = depth_normalized_foreground
            else:
                min_val, max_val = np.min(depth_map), np.max(depth_map)
                normalized_depth = 1 - ((depth_map - min_val) / (max_val - min_val))
                normalized_depth = (normalized_depth * 255.0).astype(np.uint8)


            output.append(
                {
                    "depth_map": depth_map,
                    "depth_map_normalized": normalized_depth,
                }
            )

        return output


    def post_process_pose_estimation(
        self,
        outputs,
        target_sizes: Optional[List[Tuple]] = None,
        threshold: float = 0.5,
    ):
        """
        Converts the output of [`SapiensForPoseEstimation`] into pose estimation keypoints. Only supports PyTorch.

        Args:
            outputs ([`SapiensForPoseEstimation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            keypoints: `List[np.ndarray]` of length `batch_size`, where each item is a numpy array of shape
            (num_keypoints, 2) corresponding to the target_sizes entry (if `target_sizes` is specified).
        """
        heatmaps = outputs.logits
        heatmaps = interpolate_if_needed(heatmaps, target_sizes, mode="bilinear", align_corners=False)

        output = []
        for heatmap in heatmaps:
            heatmap = heatmap.cpu().numpy()
            keypoints, scores = get_heatmap_maximum(heatmap)
            keep = scores > threshold
            keypoints = keypoints[keep]
            scores = scores[keep]

            output.append(
                {
                    "keypoints": keypoints,
                    "scores": scores,
                }
            )

        return output
