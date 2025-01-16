# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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

import base64
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import requests
from packaging import version

from .utils import (
    ExplicitEnum,
    TensorType,
    is_av_available,
    is_cv2_available,
    is_decord_available,
    is_jax_tensor,
    is_numpy_array,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_torchvision_available,
    is_vision_available,
    is_yt_dlp_available,
    logging,
    requires_backends,
    to_numpy,
)
from .utils.constants import (  # noqa: F401
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)


if is_vision_available():
    import PIL.Image
    import PIL.ImageOps

    if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
        PILImageResampling = PIL.Image.Resampling
    else:
        PILImageResampling = PIL.Image

    if is_torchvision_available():
        from torchvision import io as torchvision_io
        from torchvision.transforms import InterpolationMode

        pil_torch_interpolation_mapping = {
            PILImageResampling.NEAREST: InterpolationMode.NEAREST,
            PILImageResampling.BOX: InterpolationMode.BOX,
            PILImageResampling.BILINEAR: InterpolationMode.BILINEAR,
            PILImageResampling.HAMMING: InterpolationMode.HAMMING,
            PILImageResampling.BICUBIC: InterpolationMode.BICUBIC,
            PILImageResampling.LANCZOS: InterpolationMode.LANCZOS,
        }

if is_decord_available():
    from decord import VideoReader, cpu

if is_av_available():
    import av

if is_cv2_available():
    import cv2

if is_yt_dlp_available():
    from yt_dlp import YoutubeDL

if TYPE_CHECKING:
    if is_torch_available():
        import torch


logger = logging.get_logger(__name__)


ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
]  # noqa


VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["torch.Tensor"]],
]  # noqa


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"


class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"


class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = AnnotationFormat.COCO_DETECTION.value
    COCO_PANOPTIC = AnnotationFormat.COCO_PANOPTIC.value


AnnotationType = Dict[str, Union[int, str, List[Dict]]]


def is_pil_image(img):
    return is_vision_available() and isinstance(img, PIL.Image.Image)


class ImageType(ExplicitEnum):
    PIL = "pillow"
    TORCH = "torch"
    NUMPY = "numpy"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


def get_image_type(image):
    if is_pil_image(image):
        return ImageType.PIL
    if is_torch_tensor(image):
        return ImageType.TORCH
    if is_numpy_array(image):
        return ImageType.NUMPY
    if is_tf_tensor(image):
        return ImageType.TENSORFLOW
    if is_jax_tensor(image):
        return ImageType.JAX
    raise ValueError(f"Unrecognised image type {type(image)}")


def is_valid_image(img):
    return is_pil_image(img) or is_numpy_array(img) or is_torch_tensor(img) or is_tf_tensor(img) or is_jax_tensor(img)


def valid_images(imgs):
    # If we have an list of images, make sure every image is valid
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # If not a list of tuple, we have been given a single image or batched tensor of images
    elif not is_valid_image(imgs):
        return False
    return True


def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])
    return False


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


def make_list_of_images(images, expected_ndims: int = 3) -> List[ImageInput]:
    """
    Ensure that the input is a list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a batch of images, it is converted to a list of images.

    Args:
        images (`ImageInput`):
            Image of images to turn into a list of images.
        expected_ndims (`int`, *optional*, defaults to 3):
            Expected number of dimensions for a single input image. If the input image has a different number of
            dimensions, an error is raised.
    """
    if is_batched(images):
        return images

    # Either the input is a single image, in which case we create a list of length 1
    if isinstance(images, PIL.Image.Image):
        # PIL images are never batched
        return [images]

    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # Batch of images
            images = list(images)
        elif images.ndim == expected_ndims:
            # Single image
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {expected_ndims + 1} or {expected_ndims} dimensions, but got"
                f" {images.ndim} dimensions."
            )
        return images
    raise ValueError(
        "Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or "
        f"jax.ndarray, but got {type(images)}."
    )


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_pil_image(images):
        return [images]

    elif is_valid_image(images):
        if len(images.shape) == 4:
            return images
        elif len(images.shape) == 3:
            return [images]

    raise ValueError(f"Could not make batched video from {images}")


def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")

    if is_vision_available() and isinstance(img, PIL.Image.Image):
        return np.array(img)
    return to_numpy(img)


def infer_channel_dimension_format(
    image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        logger.warning(
            f"The channel dimension is ambiguous. Got image shape {image.shape}. Assuming channels are the first dimension."
        )
        return ChannelDimension.FIRST
    elif image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_channel_dimension_axis(
    image: np.ndarray, input_data_format: Optional[Union[ChannelDimension, str]] = None
) -> int:
    """
    Returns the channel dimension axis of the image.

    Args:
        image (`np.ndarray`):
            The image to get the channel dimension axis of.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the image. If `None`, will infer the channel dimension from the image.

    Returns:
        The channel dimension axis of the image.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    elif input_data_format == ChannelDimension.LAST:
        return image.ndim - 1
    raise ValueError(f"Unsupported data format: {input_data_format}")


def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def get_image_size_for_max_height_width(
    image_size: Tuple[int, int],
    max_height: int,
    max_width: int,
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image and the maximum allowed height and width. Keep aspect ratio.
    Important, even if image_height < max_height and image_width < max_width, the image will be resized
    to at least one of the edges be equal to max_height or max_width.

    For example:
        - input_size: (100, 200), max_height: 50, max_width: 50 -> output_size: (25, 50)
        - input_size: (100, 200), max_height: 200, max_width: 500 -> output_size: (200, 400)

    Args:
        image_size (`Tuple[int, int]`):
            The image to resize.
        max_height (`int`):
            The maximum allowed height.
        max_width (`int`):
            The maximum allowed width.
    """
    height, width = image_size
    height_scale = max_height / height
    width_scale = max_width / width
    min_scale = min(height_scale, width_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)
    return new_height, new_width


def is_valid_annotation_coco_detection(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "annotations" in annotation
        and isinstance(annotation["annotations"], (list, tuple))
        and (
            # an image can have no annotations
            len(annotation["annotations"]) == 0 or isinstance(annotation["annotations"][0], dict)
        )
    ):
        return True
    return False


def is_valid_annotation_coco_panoptic(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if (
        isinstance(annotation, dict)
        and "image_id" in annotation
        and "segments_info" in annotation
        and "file_name" in annotation
        and isinstance(annotation["segments_info"], (list, tuple))
        and (
            # an image can have no segments
            len(annotation["segments_info"]) == 0 or isinstance(annotation["segments_info"][0], dict)
        )
    ):
        return True
    return False


def valid_coco_detection_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    return all(is_valid_annotation_coco_detection(ann) for ann in annotations)


def valid_coco_panoptic_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    return all(is_valid_annotation_coco_panoptic(ann) for ann in annotations)


def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None) -> "PIL.Image.Image":
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    requires_backends(load_image, ["vision"])
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            image = PIL.Image.open(BytesIO(requests.get(image, timeout=timeout).content))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # Try to load as base64
            try:
                b64 = base64.decodebytes(image.encode())
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise TypeError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def get_uniform_frame_indices(total_num_frames: int, num_frames: Optional[int] = None):
    """
    Creates a numpy array for uniform sampling of `num_frame` frames from `total_num_frames`
    when loading a video.

    Args:
        total_num_frames (`int`):
            Total number of frames that a video has.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of frame indices that will be sampled.
    """
    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    else:
        indices = np.arange(0, total_num_frames).astype(int)
    return indices


def read_video_opencv(video_path: str, num_frames: Optional[int] = None):
    """
    Decode the video with open-cv decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    video = cv2.VideoCapture(video_path)
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = get_uniform_frame_indices(total_num_frames, num_frames=num_frames)

    index = 0
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if index in indices:
            height, width, channel = frame.shape
            frames.append(frame[0:height, 0:width, 0:channel])
        if success:
            index += 1
        if index >= total_num_frames:
            break

    video.release()
    return np.stack(frames)


def read_video_decord(video_path: str, num_frames: Optional[int] = None):
    """
    Decode the video with Decord decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    vr = VideoReader(uri=video_path, ctx=cpu(0))  # decord has problems with gpu
    indices = get_uniform_frame_indices(total_num_frames=len(vr), num_frames=num_frames)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def read_video_pyav(video_path: str, num_frames: Optional[int] = None):
    """
    Decode the video with PyAV decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    container = av.open(video_path)

    # sample uniformly "num_frames" frames from the video
    total_num_frames = container.streams.video[0].frames
    indices = get_uniform_frame_indices(total_num_frames, num_frames=num_frames)

    frames = []
    container.seek(0)
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= 0 and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def read_video_torchvision(video_path: str, num_frames: Optional[int] = None):
    """
    Decode the video with torchvision decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    video, _, info = torchvision_io.read_video(
        video_path,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    )

    if num_frames is not None:
        idx = torch.linspace(0, video.size(0) - 1, num_frames, dtype=torch.int64)
        return video[idx]

    return video


VIDEO_DECODERS = {
    "decord": read_video_decord,
    "opencv": read_video_opencv,
    "pyav": read_video_pyav,
    "torchvision": read_video_torchvision,
}


def load_video(video: Union[str, "VideoInput"], num_frames: Optional[int] = None, backend: str = "opencv") -> np.array:
    """
    Loads `video` to a numpy array.

    Args:
        video (`str` or `VideoInput`):
            The video to convert to the numpy array format. Can be a link to video or local path.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not passed, the whole video is loaded.
        backend (`str`, *optional*, defaults to `"opencv"`):
            The backend to use when loading the video. Can be any of ["decord", "pyav", "opencv", "torchvision"]. Defaults to "opencv".

    Returns:
        `np.array`: A numpy array of shape (num_frames, channels, height, width).
    """
    if video.startswith("https://www.youtube.com") or video.startswith("http://www.youtube.com"):
        if not is_yt_dlp_available():
            raise ImportError("To load a video from YouTube url you have  to install `yt_dlp` first.")
        buffer = BytesIO()
        with redirect_stdout(buffer), YoutubeDL() as f:
            f.download([video])
        bytes_obj = buffer.getvalue()
        file_obj = BytesIO(bytes_obj)
    elif video.startswith("http://") or video.startswith("https://"):
        file_obj = BytesIO(requests.get(video).content)
    elif os.path.isfile(video):
        file_obj = video
    elif is_valid_image(video) or (isinstance(video, (list, tuple) and is_valid_image(video[0]))):
        file_obj = None
    else:
        raise TypeError("Incorrect format used for video. Should be an url linking to an video or a local path.")

    # can also load with decord, but not cv2/torchvision
    # both will fail in case of url links
    video_is_url = video.startswith("http://") or video.startswith("https://")
    if video_is_url and backend in ["opencv", "torchvision"]:
        raise ValueError(
            "If you are trying to load a video from URL, you can decode the video only with `pyav` or `decord` as backend"
        )

    if file_obj is None:
        return video

    if (
        (not is_decord_available() and backend == "decord")
        or (not is_av_available() and backend == "pyav")
        or (not is_cv2_available() and backend == "opencv")
        or (not is_torchvision_available() and backend == "torchvision")
    ):
        raise ImportError(
            f"You chose backend={backend} for loading the video but the required library is not found in your environment "
            f"Make sure to install {backend} before loading the video."
        )

    video_decoder = VIDEO_DECODERS[backend]
    video = video_decoder(file_obj)
    return video


def load_images(
    images: Union[List, Tuple, str, "PIL.Image.Image"], timeout: Optional[float] = None
) -> Union["PIL.Image.Image", List["PIL.Image.Image"], List[List["PIL.Image.Image"]]]:
    """Loads images, handling different levels of nesting.

    Args:
      images: A single image, a list of images, or a list of lists of images to load.
      timeout: Timeout for loading images.

    Returns:
      A single image, a list of images, a list of lists of images.
    """
    if isinstance(images, (list, tuple)):
        if len(images) and isinstance(images[0], (list, tuple)):
            return [[load_image(image, timeout=timeout) for image in image_group] for image_group in images]
        else:
            return [load_image(image, timeout=timeout) for image in images]
    else:
        return load_image(images, timeout=timeout)


def validate_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_pad: Optional[bool] = None,
    size_divisibility: Optional[int] = None,
    do_center_crop: Optional[bool] = None,
    crop_size: Optional[Dict[str, int]] = None,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: Optional["PILImageResampling"] = None,
):
    """
    Checks validity of typically used arguments in an `ImageProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    Many incompatibilities are model-specific. `do_pad` sometimes needs `size_divisor`,
    sometimes `size_divisibility`, and sometimes `size`. New models and processors added should follow
    existing arguments when possible.

    """
    if do_rescale and rescale_factor is None:
        raise ValueError("`rescale_factor` must be specified if `do_rescale` is `True`.")

    if do_pad and size_divisibility is None:
        # Here, size_divisor might be passed as the value of size
        raise ValueError(
            "Depending on the model, `size_divisibility`, `size_divisor`, `pad_size` or `size` must be specified if `do_pad` is `True`."
        )

    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError("`image_mean` and `image_std` must both be specified if `do_normalize` is `True`.")

    if do_center_crop and crop_size is None:
        raise ValueError("`crop_size` must be specified if `do_center_crop` is `True`.")

    if do_resize and (size is None or resample is None):
        raise ValueError("`size` and `resample` must be specified if `do_resize` is `True`.")


def validate_fast_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_pad: Optional[bool] = None,
    size_divisibility: Optional[int] = None,
    do_center_crop: Optional[bool] = None,
    crop_size: Optional[Dict[str, int]] = None,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: Optional["PILImageResampling"] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
):
    """
    Checks validity of typically used arguments in an `ImageProcessorFast` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    """
    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_pad=do_pad,
        size_divisibility=size_divisibility,
        do_center_crop=do_center_crop,
        crop_size=crop_size,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )
    # Extra checks for ImageProcessorFast
    if return_tensors is not None and return_tensors != "pt":
        raise ValueError("Only returning PyTorch tensors is currently supported.")

    if data_format != ChannelDimension.FIRST:
        raise ValueError("Only channel first data format is currently supported.")


# In the future we can add a TF implementation here when we have TF models.
class ImageFeatureExtractionMixin:
    """
    Mixin that contain utilities for preparing image features.
    """

    def _ensure_format_supported(self, image):
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
                "`torch.Tensor` are."
            )

    def to_pil_image(self, image, rescale=None):
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        self._ensure_format_supported(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def convert_rgb(self, image):
        """
        Converts `PIL.Image.Image` to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The image to convert.
        """
        self._ensure_format_supported(image)
        if not isinstance(image, PIL.Image.Image):
            return image

        return image.convert("RGB")

    def rescale(self, image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
        """
        Rescale a numpy image by scale amount
        """
        self._ensure_format_supported(image)
        return image * scale

    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
            channel_first (`bool`, *optional*, defaults to `True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if is_torch_tensor(image):
            image = image.numpy()

        rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

        if rescale:
            image = self.rescale(image.astype(np.float32), 1 / 255.0)

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def expand_dims(self, image):
        """
        Expands 2-dimensional `image` to 3 dimensions.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to expand.
        """
        self._ensure_format_supported(image)

        # Do nothing if PIL image
        if isinstance(image, PIL.Image.Image):
            return image

        if is_torch_tensor(image):
            image = image.unsqueeze(0)
        else:
            image = np.expand_dims(image, axis=0)
        return image

    def normalize(self, image, mean, std, rescale=False):
        """
        Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
        if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to normalize.
            mean (`List[float]` or `np.ndarray` or `torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (`List[float]` or `np.ndarray` or `torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
            rescale (`bool`, *optional*, defaults to `False`):
                Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
                happen automatically.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image, rescale=True)
        # If the input image is a PIL image, it automatically gets rescaled. If it's another
        # type it may need rescaling.
        elif rescale:
            if isinstance(image, np.ndarray):
                image = self.rescale(image.astype(np.float32), 1 / 255.0)
            elif is_torch_tensor(image):
                image = self.rescale(image.float(), 1 / 255.0)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        elif is_torch_tensor(image):
            import torch

            if not isinstance(mean, torch.Tensor):
                if isinstance(mean, np.ndarray):
                    mean = torch.from_numpy(mean)
                else:
                    mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std)
                else:
                    std = torch.tensor(std)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def resize(self, image, size, resample=None, default_to_square=True, max_size=None):
        """
        Resizes `image`. Enforces conversion of input to PIL.Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to resize.
            size (`int` or `Tuple[int, int]`):
                The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be
                matched to this.

                If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
                `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to
                this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                The filter to user for resampling.
            default_to_square (`bool`, *optional*, defaults to `True`):
                How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a
                square (`size`,`size`). If set to `False`, will replicate
                [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
                with support for resizing only the smallest edge and providing an optional `max_size`.
            max_size (`int`, *optional*, defaults to `None`):
                The maximum allowed for the longer edge of the resized image: if the longer edge of the image is
                greater than `max_size` after being resized according to `size`, then the image is resized again so
                that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller
                edge may be shorter than `size`. Only used if `default_to_square` is `False`.

        Returns:
            image: A resized `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PILImageResampling.BILINEAR

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        if isinstance(size, list):
            size = tuple(size)

        if isinstance(size, int) or len(size) == 1:
            if default_to_square:
                size = (size, size) if isinstance(size, int) else (size[0], size[0])
            else:
                width, height = image.size
                # specified size only for the smallest edge
                short, long = (width, height) if width <= height else (height, width)
                requested_new_short = size if isinstance(size, int) else size[0]

                if short == requested_new_short:
                    return image

                new_short, new_long = requested_new_short, int(requested_new_short * long / short)

                if max_size is not None:
                    if max_size <= requested_new_short:
                        raise ValueError(
                            f"max_size = {max_size} must be strictly greater than the requested "
                            f"size for the smaller edge size = {size}"
                        )
                    if new_long > max_size:
                        new_short, new_long = int(max_size * new_short / new_long), max_size

                size = (new_short, new_long) if width <= height else (new_long, new_short)

        return image.resize(size, resample=resample)

    def center_crop(self, image, size):
        """
        Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
        size given, it will be padded (so the returned result has the size asked).

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape (n_channels, height, width) or (height, width, n_channels)):
                The image to resize.
            size (`int` or `Tuple[int, int]`):
                The size to which crop the image.

        Returns:
            new_image: A center cropped `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape: (n_channels,
            height, width).
        """
        self._ensure_format_supported(image)

        if not isinstance(size, tuple):
            size = (size, size)

        # PIL Image.size is (width, height) but NumPy array and torch Tensors have (height, width)
        if is_torch_tensor(image) or isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = self.expand_dims(image)
            image_shape = image.shape[1:] if image.shape[0] in [1, 3] else image.shape[:2]
        else:
            image_shape = (image.size[1], image.size[0])

        top = (image_shape[0] - size[0]) // 2
        bottom = top + size[0]  # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
        left = (image_shape[1] - size[1]) // 2
        right = left + size[1]  # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.

        # For PIL Images we have a method to crop directly.
        if isinstance(image, PIL.Image.Image):
            return image.crop((left, top, right, bottom))

        # Check if image is in (n_channels, height, width) or (height, width, n_channels) format
        channel_first = True if image.shape[0] in [1, 3] else False

        # Transpose (height, width, n_channels) format images
        if not channel_first:
            if isinstance(image, np.ndarray):
                image = image.transpose(2, 0, 1)
            if is_torch_tensor(image):
                image = image.permute(2, 0, 1)

        # Check if cropped area is within image boundaries
        if top >= 0 and bottom <= image_shape[0] and left >= 0 and right <= image_shape[1]:
            return image[..., top:bottom, left:right]

        # Otherwise, we may need to pad if the image is too small. Oh joy...
        new_shape = image.shape[:-2] + (max(size[0], image_shape[0]), max(size[1], image_shape[1]))
        if isinstance(image, np.ndarray):
            new_image = np.zeros_like(image, shape=new_shape)
        elif is_torch_tensor(image):
            new_image = image.new_zeros(new_shape)

        top_pad = (new_shape[-2] - image_shape[0]) // 2
        bottom_pad = top_pad + image_shape[0]
        left_pad = (new_shape[-1] - image_shape[1]) // 2
        right_pad = left_pad + image_shape[1]
        new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        new_image = new_image[
            ..., max(0, top) : min(new_image.shape[-2], bottom), max(0, left) : min(new_image.shape[-1], right)
        ]

        return new_image

    def flip_channel_order(self, image):
        """
        Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
        `image` to a NumPy array if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should
                be first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        return image[::-1, :, :]

    def rotate(self, image, angle, resample=None, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
        counter clockwise around its centre.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
                rotating.

        Returns:
            image: A rotated `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PIL.Image.NEAREST

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.rotate(
            angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
        )


def validate_annotations(
    annotation_format: AnnotationFormat,
    supported_annotation_formats: Tuple[AnnotationFormat, ...],
    annotations: List[Dict],
) -> None:
    if annotation_format not in supported_annotation_formats:
        raise ValueError(f"Unsupported annotation format: {format} must be one of {supported_annotation_formats}")

    if annotation_format is AnnotationFormat.COCO_DETECTION:
        if not valid_coco_detection_annotations(annotations):
            raise ValueError(
                "Invalid COCO detection annotations. Annotations must a dict (single image) or list of dicts "
                "(batch of images) with the following keys: `image_id` and `annotations`, with the latter "
                "being a list of annotations in the COCO format."
            )

    if annotation_format is AnnotationFormat.COCO_PANOPTIC:
        if not valid_coco_panoptic_annotations(annotations):
            raise ValueError(
                "Invalid COCO panoptic annotations. Annotations must a dict (single image) or list of dicts "
                "(batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with "
                "the latter being a list of annotations in the COCO format."
            )


def validate_kwargs(valid_processor_keys: List[str], captured_kwargs: List[str]):
    unused_keys = set(captured_kwargs).difference(set(valid_processor_keys))
    if unused_keys:
        unused_key_str = ", ".join(unused_keys)
        # TODO raise a warning here instead of simply logging?
        logger.warning(f"Unused or unrecognized kwargs: {unused_key_str}.")


@dataclass(frozen=True)
class SizeDict:
    """
    Hashable dictionary to store image size information.
    """

    height: int = None
    width: int = None
    longest_edge: int = None
    shortest_edge: int = None
    max_height: int = None
    max_width: int = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")
