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
"""Image processor class for LayoutLMv2."""

from typing import Dict, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import flip_channel_order, resize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
    is_pytesseract_available,
    is_vision_available,
    logging,
    requires_backends,
)


if is_vision_available():
    import PIL

# soft dependency
if is_pytesseract_available():
    import pytesseract

logger = logging.get_logger(__name__)


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_tesseract(
    image: np.ndarray,
    lang: Optional[str],
    tesseract_config: Optional[str] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    tesseract_config = tesseract_config if tesseract_config is not None else ""

    # apply OCR
    pil_image = to_pil_image(image, input_data_format=input_data_format)
    image_width, image_height = pil_image.size
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    return words, normalized_boxes


class LayoutLMv2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LayoutLMv2 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to `(size["height"], size["width"])`. Can be
            overridden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
            `apply_ocr` in `preprocess`.
        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used. Can be overridden by `ocr_lang` in `preprocess`.
        tesseract_config (`str`, *optional*, defaults to `""`):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'. Can be overridden by `tesseract_config` in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        apply_ocr: bool = True,
        ocr_lang: Optional[str] = None,
        tesseract_config: Optional[str] = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.apply_ocr = apply_ocr
        self.ocr_lang = ocr_lang
        self.tesseract_config = tesseract_config

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize
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
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        apply_ocr: bool = None,
        ocr_lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Desired size of the output image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image` resampling
                filter. Only has an effect if `do_resize` is set to `True`.
            apply_ocr (`bool`, *optional*, defaults to `self.apply_ocr`):
                Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes.
            ocr_lang (`str`, *optional*, defaults to `self.ocr_lang`):
                The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
                used.
            tesseract_config (`str`, *optional*, defaults to `self.tesseract_config`):
                Any additional custom configuration flags that are forwarded to the `config` parameter when calling
                Tesseract.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size)
        resample = resample if resample is not None else self.resample
        apply_ocr = apply_ocr if apply_ocr is not None else self.apply_ocr
        ocr_lang = ocr_lang if ocr_lang is not None else self.ocr_lang
        tesseract_config = tesseract_config if tesseract_config is not None else self.tesseract_config

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if apply_ocr:
            requires_backends(self, "pytesseract")
            words_batch = []
            boxes_batch = []
            for image in images:
                words, boxes = apply_tesseract(image, ocr_lang, tesseract_config, input_data_format=input_data_format)
                words_batch.append(words)
                boxes_batch.append(boxes)

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        # flip color channels from RGB to BGR (as Detectron2 requires this)
        images = [flip_channel_order(image, input_data_format=input_data_format) for image in images]
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

        if apply_ocr:
            data["words"] = words_batch
            data["boxes"] = boxes_batch
        return data


__all__ = ["LayoutLMv2ImageProcessor"]
