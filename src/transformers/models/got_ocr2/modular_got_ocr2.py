# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from transformers.models.blip.image_processing_blip import BlipImageProcessor
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)

from ...activations import ACT2FN
from ...cache_utils import StaticCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import (
    _rescale_for_pil_conversion,
    to_channel_dimension_format,
    to_pil_image,
)
from ...image_utils import ChannelDimension, ImageInput
from ...modeling_outputs import CausalLMOutputWithPast
from ...utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    is_vision_available,
    logging,
    replace_return_docstrings,
)


if is_vision_available():
    import PIL

    from ...image_utils import load_images

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GotOcr2Config"


class GotOcr2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GotOcr2VisionModel`]. It is used to instantiate a GOT_OCR2
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM ViT-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        output_channels (`int`, *optional*, defaults to 256):
            Dimensionality of the output channels in the Patch Encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        image_size (`int`, *optional*, defaults to 1024):
            Expected resolution. Target size of the resized input image.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to be extracted from the input image.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string)
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to query, key, value projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of mlp hidden dim to embedding dim.
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embedding.
        use_rel_pos (`bool`, *optional*, defaults to `True`):
            Whether to use relative position embedding.
        window_size (`int`, *optional*, defaults to 14):
            Window size for relative position.
        global_attn_indexes (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
            The indexes of the global attention layers.
        mlp_dim (`int`, *optional*):
            The dimensionality of the MLP layer in the Transformer encoder. If `None`, defaults to `mlp_ratio *
            hidden_size`.
    """

    def __init__(
        self,
        hidden_size=768,
        output_channels=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        mlp_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.mlp_dim = mlp_dim


class GotOcr2Config(Qwen2VLConfig):
    pass


class GotOcr2TextKwargs(TextKwargs, total=False):
    format: Optional[bool]


class GotOcr2ImagesKwargs(ImagesKwargs, total=False):
    box: Optional[Union[List, Tuple[float, float], Tuple[float, float, float, float]]]
    color: Optional[str]
    num_image_tokens: Optional[int]
    multi_page: Optional[bool]
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class GotOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: GotOcr2TextKwargs
    images_kwargs: GotOcr2ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "format": False,
        },
        "images_kwargs": {
            "num_image_tokens": 256,
            "multi_page": False,
            "crop_to_patches": False,
            "min_patches": 1,
            "max_patches": 12,
        },
    }


def preprocess_box_annotation(box: Union[List, Tuple], image_size: Tuple[int, int]) -> List:
    """
    Convert box annotation to the format [x1, y1, x2, y2] in the range [0, 1000].
    """
    width, height = image_size
    if len(box) == 4:
        box[0] = int(box[0] / width * 1000)
        box[1] = int(box[1] / height * 1000)
        box[2] = int(box[2] / width * 1000)
        box[3] = int(box[3] / height * 1000)
    else:
        raise ValueError("Box must be a list or tuple of lists in the form [x1, y1, x2, y2].")

    return list(box)


def find_best_patches_grid(
    original_image_size: dict,
    target_patch_size: dict,
    min_patches: int,
    max_patches: int,
) -> Tuple[int, int]:
    """
    Given a minimum and maximum number of patches, find the patches grid with the closest aspect ratio to the
    original image aspect ratio.
    In case of tie-breaking condition when two grids have the same aspect ratio difference, we favor the grids with
    more patches, until the area covered by the patches is more than twice the target area, in order to avoid unnecessarily
    excessive patching.
    """
    # compute possible patches grids
    target_patches_grids = {
        (i, j)
        for i in range(1, max_patches + 1)
        for j in range(1, max_patches + 1)
        if i * j <= max_patches and i * j >= min_patches
    }
    target_patches_grids = sorted(target_patches_grids, key=lambda x: x[0] * x[1])

    # find the grid with the best aspect ratio
    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    original_width, original_height = original_image_size["width"], original_image_size["height"]
    aspect_ratio = original_width / original_height
    area = original_width * original_height
    for grid in target_patches_grids:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            # if the aspect ratio difference is the same, we favor the grid with more patches
            # until the area covered by the patches is more than twice the original image area
            if area > 0.5 * target_patch_size["width"] * target_patch_size["height"] * grid[0] * grid[1]:
                best_grid = grid

    return best_grid


class GotOcr2ImageProcessor(BlipImageProcessor):
    def crop_image_to_patches(
        self,
        image: ImageInput,
        min_patches: int,
        max_patches: int,
        use_thumbnail: bool = True,
        patch_size: Union[Tuple, int, dict] = None,
        return_numpy: bool = False,
        data_format: ChannelDimension = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`):
                The image to be cropped. The image can be a PIL image, NumPy array or PyTorch tensor.
            min_patches (`int`):
                The minimum number of patches to be extracted from the image.
            max_patches (`int`):
                The maximum number of patches to be extracted from the image.
            use_thumbnail (`bool`, *optional*, defaults to `True`):
                Whether to add a thumbnail image to the list of cropped patches.
            patch_size (`int`, `Tuple[int, int]`, `dict`, *optional*):
                The size of the output patches.
            return_numpy (`bool`, *optional*, defaults to `False`):
                Whether to return the cropped images as NumPy arrays.
            data_format (`ChannelDimension`, *optional*):
                The format of the image data. If `None`, the format is inferred from the input image.

        Returns:
            List[`PIL.Image.Image`] or List[np.ndarray]: The list of cropped images.
        """
        patch_size = patch_size if patch_size is not None else self.size
        patch_size = get_size_dict(patch_size, default_to_square=True)
        original_size = get_size_dict(image.size, height_width_order=False)
        do_rescale = False
        if not isinstance(image, PIL.Image.Image):
            do_rescale = _rescale_for_pil_conversion(image)
            image = to_pil_image(image, do_rescale=do_rescale)

        # find the closest aspect ratio to the target
        num_columns, num_rows = find_best_patches_grid(original_size, patch_size, min_patches, max_patches)

        # calculate the target width and height
        patch_size_width, patch_size_height = patch_size["width"], patch_size["height"]
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = image.resize((target_width, target_height))

        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            # split the image
            patch_image = resized_image.crop(box)
            processed_images.append(patch_image)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((patch_size_width, patch_size_height))
            processed_images.append(thumbnail_img)

        if return_numpy:
            processed_images_numpy = []
            for processed_image in processed_images:
                processed_image = np.array(processed_image)
                # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
                # so we need to add it back if necessary.
                processed_image = (
                    np.expand_dims(processed_image, axis=-1) if processed_image.ndim == 2 else processed_image
                )
                # The image is always in channels last format after converting from a PIL image
                if data_format is not None:
                    processed_image = to_channel_dimension_format(
                        processed_image, data_format, input_channel_dim=ChannelDimension.LAST
                    )
                # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
                # rescale it back to the original range.
                processed_image = self.rescale(processed_image, 1 / 255) if do_rescale else processed_image
                processed_images_numpy.append(processed_image)
            processed_images = processed_images_numpy

        return processed_images


class GotOcr2Processor(ProcessorMixin):
    r"""
    Constructs a GotOcr2 processor which wraps a [`GotOcr2ImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~GotOcr2Processor.__call__`] and [`~GotOcr2Processor.decode`] for more information.
    Args:
        image_processor ([`GotOcr2ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "GotOcr2ImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.message_start_token = "<|im_start|>"
        self.message_end_token = "<|im_end|>"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_pad_token = "<imgpad>"
        self.system_query = "system\nYou should follow the instructions carefully and explain your answers in detail."

    def _check_call_arguments(self, images, box, color, multi_page, crop_to_patches):
        if images is None:
            raise ValueError("Images are required to be passed to the processor.")

        if not isinstance(box, (list, tuple)):
            raise ValueError("Box must be a list or tuple of lists in the form [x1, y1, x2, y2].")

        if multi_page or crop_to_patches:
            if multi_page and crop_to_patches:
                raise ValueError("Cannot set both `multi_page` and `crop_to_patches` to `True`.")
            if box[0] is not None or color is not None:
                raise ValueError("Cannot pass `box` or `color` with multi-page inference.")

        if box[0] is not None and color is not None:
            raise ValueError("Both `box` and `color` cannot be set at the same time.")

    def _make_list_of_inputs(self, images, text, box, color, multi_page):
        if not isinstance(images, (list, tuple)):
            if multi_page:
                logger.warning("Multi-page inference is enabled but only one image is passed.")
            images = [images]
        elif isinstance(images[0], (list, tuple)) and not multi_page:
            raise ValueError("Nested images are only supported with `multi_page` set to `True`.")
        elif not isinstance(images[0], (list, tuple)) and multi_page:
            images = [images]

        if text is not None:
            if not isinstance(text, (list, tuple)):
                text = [text]
            if len(text) != len(images):
                raise ValueError("The number of `text` must match the number of images.")

        if not isinstance(box[0], (list, tuple)):
            # Use the same box for all images
            box = [box for _ in range(len(images))]
        if not isinstance(color, (list, tuple)):
            color = [color for _ in range(len(images))]
        if len(box) != len(images):
            raise ValueError("The number of `box` must match the number of images.")
        if len(color) != len(images):
            raise ValueError("The number of `color` must match the number of images.")

        return images, text, box, color

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[GotOcr2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text if `text`
        is not `None`, otherwise encode default OCR queries which depends on the `format`, `box`, `color`, `multi_page` and
        `crop_to_patches` arguments. To prepare the vision inputs, this method forwards the `images` and `kwrags` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            format (`bool`, *optional*):
                If set, will add the format token to the query, and the model will return the OCR result with formatting.
            box (`List[float]`, `List[Tuple[float, float]]`, `List[Tuple[float, float, float, float]]`, *optional*):
                The box annotation to be added to the query. If a list of floats or a tuple of floats is provided, it
                will be interpreted as [x1, y1, x2, y2]. If a list of tuples is provided, each tuple should be in the
                form (x1, y1, x2, y2).
            color (`str`, *optional*):
                The color annotation to be added to the query. The model will return the OCR result within the box with
                the specified color.
            multi_page (`bool`, *optional*):
                If set, will enable multi-page inference. The model will return the OCR result across multiple pages.
            crop_to_patches (`bool`, *optional*):
                If set, will crop the image to patches. The model will return the OCR result upon the patch reference.
            min_patches (`int`, *optional*):
                The minimum number of patches to be cropped from the image. Only used when `crop_to_patches` is set to
                `True`.
            max_patches (`int`, *optional*):
                The maximum number of patches to be cropped from the image. Only used when `crop_to_patches` is set to
                `True`.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            GotOcr2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        format_output = output_kwargs["text_kwargs"].pop("format")
        num_image_tokens = output_kwargs["images_kwargs"].pop("num_image_tokens")
        box = output_kwargs["images_kwargs"].pop("box", [None])
        color = output_kwargs["images_kwargs"].pop("color", None)
        multi_page = output_kwargs["images_kwargs"].pop("multi_page")
        crop_to_patches = output_kwargs["images_kwargs"].pop("crop_to_patches")
        min_patches = output_kwargs["images_kwargs"].pop("min_patches")
        max_patches = output_kwargs["images_kwargs"].pop("max_patches")

        self._check_call_arguments(images, box, color, multi_page, crop_to_patches)
        images, text, box, color = self._make_list_of_inputs(images, text, box, color, multi_page)

        # Load images as we need to know the image size
        images = load_images(images)
        if text is None:
            text = []
            for index, (image_group, box_single, color_single) in enumerate(zip(images, box, color)):
                if crop_to_patches:
                    image_group = self.image_processor.crop_image_to_patches(
                        image_group,
                        patch_size=output_kwargs["images_kwargs"].get("size"),
                        min_patches=min_patches,
                        max_patches=max_patches,
                    )
                    images[index] = image_group
                num_images = len(image_group) if (multi_page or crop_to_patches) else 1
                if box_single[0] is not None:
                    box_single = preprocess_box_annotation(box_single, image_group.size)
                query = (
                    f"{f'[{color_single}] ' if color_single is not None else ''}"
                    f"{str(box_single) if box_single[0] is not None else ''} "
                    "OCR"
                    f"{' with format' if format_output else ''}"
                    f"{' across multi pages' if multi_page else ''}"
                    f"{' upon the patch reference' if crop_to_patches else ''}"
                    ": "
                )
                prompt = (
                    self.message_start_token
                    + self.system_query
                    + self.message_end_token
                    + self.message_start_token
                    + "user\n"
                    + self.img_start_token
                    + self.img_pad_token * num_image_tokens * num_images
                    + self.img_end_token
                    + "\n"
                    + query
                    + self.message_end_token
                    + self.message_start_token
                    + "assistant\n"
                )
                text.append(prompt)
        elif crop_to_patches:
            for index, (image_group, box_single, color_single) in enumerate(zip(images, box, color)):
                image_group = self.image_processor.crop_image_to_patches(
                    image_group,
                    patch_size=output_kwargs["images_kwargs"].get("size"),
                    min_patches=min_patches,
                    max_patches=max_patches,
                )
                images[index] = image_group

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if multi_page or crop_to_patches:
            # flatten images
            images = [image for image_group in images for image in image_group]
        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)


class GotOcr2MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dim = config.mlp_dim if config.mlp_dim is not None else int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, config.hidden_size)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GotOcr2VisionAdapter(nn.Module):
    def __init__(self, language_hidden_size: int, vision_output_channels: int):
        super().__init__()
        self.conv_upsampler1 = nn.Conv2d(
            vision_output_channels, vision_output_channels * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_upsampler2 = nn.Conv2d(
            vision_output_channels * 2, language_hidden_size, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.multimodal_projector = nn.Linear(language_hidden_size, language_hidden_size)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv_upsampler1(vision_embeddings)
        hidden_state = self.conv_upsampler2(hidden_state)
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        hidden_state = self.multimodal_projector(hidden_state)
        return hidden_state


class GotOcr2VisionEncoder(SamVisionEncoder):
    pass


class GotOcr2PreTrainedModel(Qwen2PreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GotOcr2Model(Qwen2Model):
    pass


GOT_OCR2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`GotOcr2ImageProcessor.__call__`] for details. [`GotOcr2Processor`] uses
            [`GotOcr2ImageProcessor`] for processing images.
"""


class GotOcr2ForConditionalGeneration(GotOcr2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = GotOcr2VisionEncoder(config.vision_config)
        self.model = GotOcr2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.visual_adapter = GotOcr2VisionAdapter(config.hidden_size, config.vision_config.output_channels)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        return model_kwargs

    @add_start_docstrings_to_model_forward(GOT_OCR2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, GotOcr2ForConditionalGeneration, TextStreamer

        >>> model = GotOcr2ForConditionalGeneration.from_pretrained("yonigozlan/GOT-OCR-2.0-hf").to("cuda")
        >>> processor = AutoProcessor.from_pretrained("yonigozlan/GOT-OCR-2.0-hf")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(image, return_tensors="pt", color="green").to("cuda")

        >>> # Generate
        >>> streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        >>> generate_ids = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     tokenizer = processor.tokenizer,
        ...     stop_strings='<|im_end|>',
        ...     streamer=streamer,
        ...     max_new_tokens=4096,
        ... )
        "You should keep in mind what features from the module should be used, especially
        when you're planning to sell a template."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.to(inputs_embeds.dtype)
                image_embeds = self.visual(pixel_values)
                image_embeds = self.visual_adapter(image_embeds.last_hidden_state)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0] * image_embeds.shape[1]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "cache_position": cache_position,
            }
        )
        return model_inputs


__all__ = [
    "GotOcr2VisionConfig",
    "GotOcr2Config",
    "GotOcr2Processor",
    "GotOcr2PreTrainedModel",
    "GotOcr2Model",
    "GotOcr2ForConditionalGeneration",
    "GotOcr2ImageProcessor",
]
