# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Idefics."""

from typing import Callable, Dict, List, Optional, Union

from PIL import Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available


IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]
IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Resize to image size
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        image_num_channels (`int`, *optional*, defaults to 3):
            Number of image channels.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 224,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        image_num_channels: Optional[int] = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size
        self.image_num_channels = image_num_channels
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(
        self,
        images: ImageInput,
        image_num_channels: Optional[int] = 3,
        image_size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        transform: Callable = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> TensorType:
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Resize to image size
            image_num_channels (`int`, *optional*, defaults to `self.image_num_channels`):
                Number of image channels.
            image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
                Mean to use if normalizing the image. This is a float or list of floats the length of the number of
                channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
                be overridden by the `image_mean` parameter in the `preprocess` method.
            image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
                Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
                number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
                method. Can be overridden by the `image_std` parameter in the `preprocess` method.
            transform (`Callable`, *optional*, defaults to `None`):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
                assumed - and then a preset of inference-specific transforms will be applied to the images

        Returns:
            a PyTorch tensor of the processed images

        """
        image_size = image_size if image_size is not None else self.image_size
        image_num_channels = image_num_channels if image_num_channels is not None else self.image_num_channels
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = (image_size, image_size)

        if isinstance(images, list) and len(images) == 0:
            return []

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # For training a user needs to pass their own set of transforms as a Callable.
        # For reference this is what was used in the original IDEFICS training:
        # transform = transforms.Compose([
        #     convert_to_rgb,
        #     transforms.RandomResizedCrop((size, size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=image_mean, std=image_std),
        # ])
        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in `transform` torch must be installed")
            import torch

            images = [transform(x) for x in images]
            return torch.stack(images)

        # for inference we do the exact transforms that were used to train IDEFICS
        images = [convert_to_rgb(x) for x in images]
        # further transforms expect numpy arrays
        images = [to_numpy_array(x) for x in images]
        images = [resize(x, size, resample=PILImageResampling.BICUBIC) for x in images]
        images = [self.rescale(image=image, scale=1 / 255) for image in images]
        images = [self.normalize(x, mean=image_mean, std=image_std) for x in images]
        images = [to_channel_dimension_format(x, ChannelDimension.FIRST) for x in images]
        images = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)["pixel_values"]

        return images


__all__ = ["IdeficsImageProcessor"]
