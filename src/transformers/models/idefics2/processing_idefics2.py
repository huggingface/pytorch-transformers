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
"""
Processor class for IDEFICS2.
"""

from itertools import accumulate
from typing import TYPE_CHECKING, List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import AddedToken, TextInput
from ...utils import logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class Idefics2ImagesKwargs(ImagesKwargs, total=False):
    image_seq_len: Optional[int]


class Idefics2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Idefics2ImagesKwargs

    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
        "images_kwargs": {},
    }


class Idefics2Processor(ProcessorMixin):
    r"""
    Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`Idefics2ImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics2ImageProcessor`):
            An instance of [`Idefics2ImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            config.perceiver_config.resampler_n_latents value for the model used.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["image_seq_len", "chat_template"]
    image_processor_class = "Idefics2ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 64, chat_template: str = None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        if not hasattr(tokenizer, "image_token"):
            self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True)
            self.image_token = AddedToken("<image>", normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [self.fake_image_token, self.image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
        else:
            self.fake_image_token = tokenizer.image_boundary_token
            self.image_token = tokenizer.image_token

        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [self.end_of_utterance_token]})
        self.image_seq_len = image_seq_len

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _extract_images_from_prompts(self, prompts):
        prompt_images = []
        for prompt in prompts:
            images = []
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))
            prompt_images.append(images)
        return prompt_images

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Idefics2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Processes the input prompts and returns a BatchEncoding.

        Example:

        ```python
        >>> import requests
        >>> from transformers import Idefics2Processor
        >>> from transformers.image_utils import load_image

        >>> processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b", image_seq_len=2)
        >>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [[image1], [image2]]

        >>> text = [
        ...     "<image>In this image, we see",
        ...     "bla bla bla<image>",
        ... ]
        >>> outputs = processor(images=images, text=text, return_tensors="pt", padding=True)
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        ['<s><fake_token_around_image><image><image><fake_token_around_image> In this image, we see', '<s> bla bla bla<fake_token_around_image><image><image><fake_token_around_image>']
        ```

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `List[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.

        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            Idefics2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_seq_len = output_kwargs["images_kwargs"].pop("image_seq_len", None)
        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        n_images_in_text = []
        inputs = BatchFeature()

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            fake_image_token = self.fake_image_token.content
            image_token = self.image_token.content
            image_str = f"{fake_image_token}{image_token * image_seq_len}{fake_image_token}"

            if self.image_processor.do_image_splitting:
                # A single image token is split into 4 patches + 1 original image
                image_str = image_str * 5

            prompt_strings = []
            for sample in text:
                n_images_in_text.append(sample.count(image_token))
                sample = sample.replace(image_token, image_str)
                # Remove any double fake tokens if images are adjacent
                sample = sample.replace(f"{fake_image_token}{fake_image_token}", f"{fake_image_token}")
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, list) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {image_token} tokens and {len(images)} images."
                        )
                    # Reorganize the images to match the prompts
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    images = [images]

            elif (
                not isinstance(images, list)
                and not isinstance(images[0], list)
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )

            n_images_in_images = [len(sample) for sample in images]
            if text is not None and not n_images_in_images == n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images  {n_images_in_images} should be the same."
                )

            # Load images if they are URLs
            images = [[load_image(im) for im in sample] for sample in images]
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Idefics2Processor"]
