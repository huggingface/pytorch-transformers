# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

"""Processor class for Mllama."""

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)


class MllamaImagesKwargs(ImagesKwargs, total=False):
    max_image_tiles: Optional[int]


class MllamaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MllamaImagesKwargs

    _defaults = {
        "image_kwargs": {
            "max_image_tiles": 4,
        },
    }


def get_cross_attention_token_mask(input_ids: List[int], image_token_id: int) -> List[List[int]]:
    """
    Generate a cross-attention token mask for image tokens in the input sequence.

    This function identifies the positions of image tokens in the input sequence and creates
    a mask that defines which subsequent tokens each image token should attend to.

    Args:
        input_ids (List[int]): A list of token ids representing the input sequence.
        image_token_id (int): The id of the token used to represent images in the sequence.

    Returns:
        List[List[int]]: A list of [start, end] pairs, where each pair represents the range
        of tokens an image token should attend to.

    Notes:
        - If no image tokens are present, an empty list is returned.
        - For a single image token, it attends to all subsequent tokens until the end of the sequence.
        - For multiple image tokens, each attends to tokens up to the next image token or the end of the sequence.
        - Consecutive image tokens are treated as a group and attend to all subsequent tokens together.
    """

    image_token_locations = [i for i, token in enumerate(input_ids) if token == image_token_id]

    if len(image_token_locations) == 0:
        return []

    # only one image present, unmask until end of sequence
    if len(image_token_locations) == 1:
        return [[image_token_locations[0], -1]]

    vision_masks = [[loc1, loc2] for loc1, loc2 in zip(image_token_locations[:-1], image_token_locations[1:])]

    # last image will attend to all subsequent text
    vision_masks.append([image_token_locations[-1], len(input_ids)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]

    return vision_masks


def convert_sparse_cross_attention_mask_to_dense(
    cross_attention_token_mask: List[List[List[int]]],
    num_tiles: List[List[int]],
    max_num_tiles: int,
    length: int,
) -> np.ndarray:
    """
    Convert the cross attention mask indices to a cross attention mask 4D array.

    This function takes a sparse representation of cross attention masks and converts it to a dense 4D numpy array.
    The sparse representation is a nested list structure that defines attention ranges for each image in each batch item.

    Args:
        cross_attention_token_mask (List[List[List[int]]]): A nested list structure where:
            - The outer list represents the batch dimension.
            - The middle list represents different images within each batch item.
            - The inner list contains pairs of integers [start, end] representing token ranges for each image.
        num_tiles (List[List[int]]): A nested list structure specifying the number of tiles for each image in each batch item.
        max_num_tiles (int): The maximum possible number of tiles.
        length (int): The total sequence length of the input.

    Returns:
        np.ndarray: A 4D numpy array of shape (batch_size, length, max_num_images, max_num_tiles)
            The array contains `1` where attention is allowed and `0` where it is not.

    Note:
        - Special handling is done for cases where the end token is -1, which is interpreted as attending to the end of the sequence.
    """

    batch_size = len(cross_attention_token_mask)
    max_num_images = max([len(masks) for masks in cross_attention_token_mask])

    cross_attention_mask = np.zeros(
        shape=(batch_size, length, max_num_images, max_num_tiles),
        dtype=np.int64,
    )

    for sample_idx, (sample_masks, sample_num_tiles) in enumerate(zip(cross_attention_token_mask, num_tiles)):
        for mask_idx, (locations, mask_num_tiles) in enumerate(zip(sample_masks, sample_num_tiles)):
            if len(locations) == 2:
                start, end = locations
                end = min(end, length)
                if end == -1:
                    end = length
                cross_attention_mask[sample_idx, start:end, mask_idx, :mask_num_tiles] = 1
    return cross_attention_mask


def build_string_from_input(prompt: str, bos_token: str, image_token: str) -> str:
    """
    Builds a string from the input prompt by adding `bos_token` if not already present.

    Args:
        prompt (`str`):
            The input prompt string.
        bos_token (`str`):
            The beginning of sentence token to be added.
        image_token (`str`):
            The image token used to identify the start of an image sequence.

    Returns:
        str: The modified prompt string with the `bos_token` added if necessary.

    Examples:
        >>> build_string_from_input("Hello world", "<begin_of_text>", "<|image|>")
        '<begin_of_text>Hello world'

        >>> build_string_from_input("<|image|>Hello world", "<begin_of_text>", "<|image|>")
        '<|image|><begin_of_text>Hello world'

        >>> build_string_from_input("<begin_of_text>Hello world", "<begin_of_text>", "<|image|>")
        '<begin_of_text>Hello world'
    """

    if bos_token in prompt:
        return prompt

    num_image_tokens_on_start = 0
    while prompt.startswith(image_token):
        prompt = prompt[len(image_token) :]
        num_image_tokens_on_start += 1

    return f"{image_token * num_image_tokens_on_start}{bos_token}{prompt}"


class MllamaProcessor(ProcessorMixin):
    r"""
    Constructs a Mllama processor which wraps [`MllamaImageProcessor`] and
    [`PretrainedTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~MllamaProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more
    information.
    The preferred way of passing kwargs is as a dictionary per modality, see usage example below.
        ```python
        from transformers import MllamaProcessor
        from PIL import Image

        processor = MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

        processor(
            images=your_pil_image,
            text=["<|image|>If I had to write a haiku for this one"],
            images_kwargs = {"size": {"height": 448, "width": 448}},
            text_kwargs = {"padding": "right"},
            common_kwargs = {"return_tensors": "pt"},
        )
        ```

    Args:
        image_processor ([`MllamaImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`]):
            The tokenizer is a required input.

    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MllamaImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        if not hasattr(tokenizer, "image_token"):
            self.image_token = "<|image|>"
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token = tokenizer.image_token
            self.image_token_id = tokenizer.image_token_id

        self.python_token = "<|python_tag|>"
        self.python_token_id = tokenizer.convert_tokens_to_ids(self.python_token)
        self.bos_token = tokenizer.bos_token
        self.chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[MllamaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare text(s) and image(s) to be fed as input to the model. This method forwards the `text`
        arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` arguments to
        MllamaImageProcessor's [`~MllamaImageProcessor.__call__`] if `images` is not `None`. Please refer
        to the docstring of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
            TODO: add aspect_ratio_ids and aspect_ratio_mask and cross_attention_mask
        """
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        output_kwargs = self._merge_kwargs(
            MllamaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        images_kwargs = output_kwargs["images_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        data = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [t.count(self.image_token) for t in text]
            text = [build_string_from_input(text_item, self.bos_token, self.image_token) for text_item in text]
            _ = text_kwargs.pop("padding_side", None)  # hack until padding-side is an accepted kwarg by tokenizers
            encoding = self.tokenizer(text, **text_kwargs)
            data.update(encoding)

        n_images_in_images = [0]
        if images is not None:
            images = make_nested_list_of_images(images)
            n_images_in_images = [len(sample) for sample in images]

        if text is not None:
            if any(batch_img == 0 for batch_img in n_images_in_text) and not all(
                batch_img == 0 for batch_img in n_images_in_text
            ):
                raise ValueError(
                    "If a batch of text is provided, there should be either no images or at least one image per sample"
                )
            if sum(n_images_in_images) != sum(n_images_in_text):
                if images is None:
                    raise ValueError("No image were provided, but there are image tokens in the prompt")
                else:
                    raise ValueError(
                        f"The number of image token ({sum(n_images_in_text)}) should be the same as in the number of provided images ({sum(n_images_in_images)})"
                    )

        if images is not None:
            image_features = self.image_processor(images, **images_kwargs)
            num_tiles = image_features.pop("num_tiles")
            data.update(image_features)

        # Create cross attention mask
        if images is not None and text is not None:
            cross_attention_token_mask = [
                get_cross_attention_token_mask(token_ids, self.image_token_id) for token_ids in encoding["input_ids"]
            ]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=self.image_processor.max_image_tiles,
                length=max(len(input_ids) for input_ids in encoding["input_ids"]),
            )
            data["cross_attention_mask"] = cross_attention_mask

        return_tensors = common_kwargs.pop("return_tensors", None)
        batch_feature = BatchFeature(data=data, tensor_type=return_tensors)

        return batch_feature

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

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names + image_processor_input_names + ["cross_attention_mask"])


__all__ = ["MllamaProcessor"]
